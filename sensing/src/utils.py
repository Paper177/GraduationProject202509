#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
工具函数模块
包含3D->2D投影、深度图像处理、实例分割解码等辅助函数
"""

import numpy as np
import carla

from .config import EDGES

def build_projection_matrix(w, h, fov, is_behind_camera=False):
    """
    构建相机投影矩阵
    
    参数:
        w: 图像宽度
        h: 图像高度
        fov: 视场角（度）
        is_behind_camera: 是否处理相机后方的点
    
    返回:
        3x3投影矩阵
    """
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    if is_behind_camera:
        K[0, 0] = K[1, 1] = -focal
    else:
        K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
    """
    将3D世界坐标投影到2D图像平面
    
    参数:
        loc: 世界坐标点 (carla.Location)
        K: 相机内参矩阵
        w2c: 世界到相机的变换矩阵
    
    返回:
        2D图像坐标 [x, y]
    """
    point = np.array([loc.x, loc.y, loc.z, 1])
    point_camera = np.dot(w2c, point)
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
    point_img = np.dot(K, point_camera)
    if point_img[2] != 0:
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]
    return point_img[0:2]

def point_in_canvas(pos, img_h, img_w):
    """
    检查点是否在画布范围内
    
    参数:
        pos: 点坐标 [x, y]
        img_h: 图像高度
        img_w: 图像宽度
    
    返回:
        是否在画布内
    """
    return (pos[0] >= 0) and (pos[0] < img_w) and (pos[1] >= 0) and (pos[1] < img_h)

def decode_instance_segmentation(img_rgba: np.ndarray):
    """
    解码CARLA实例分割图像
    
    参数:
        img_rgba: RGBA格式的分割图像
    
    返回:
        semantic_labels: 语义标签数组
        actor_ids: 对应的Actor ID数组
    """
    semantic_labels = img_rgba[..., 2]
    actor_ids = img_rgba[..., 1].astype(np.uint16) + (img_rgba[..., 0].astype(np.uint16) << 8)
    return semantic_labels, actor_ids

def process_depth_image(image):
    """
    将CARLA深度图像转换为以米为单位的深度数组
    
    参数:
        image: CARLA深度图像对象
    
    返回:
        depth_meters: 以米为单位的深度数组
    """
    # CARLA深度图编码在 [R, G, B] 通道中
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array.astype(np.float32)  # 转换为浮点数
    
    # 将 [R, G, B] 通道合并为深度
    normalized_depth = (array[:, :, 2] + array[:, :, 1] * 256 + array[:, :, 0] * 256 * 256) / (256 * 256 * 256 - 1)
    
    # 乘以 1000.0 (摄像机远裁剪平面) 得到以米为单位的深度
    depth_meters = normalized_depth * 1000.0
    return depth_meters

def get_truth_3d_box_projection(actor, ego, camera_bp, camera, K, world_2_camera):
    """
    从CARLA真值获取3D边界框的2D投影
    
    参数:
        actor: 要投影的CARLA Actor
        ego: 主角车辆
        camera_bp: 相机蓝图
        camera: 相机Actor
        K: 相机内参矩阵
        world_2_camera: 世界到相机的变换矩阵
    
    返回:
        投影后的2D线条列表 [(x1, y1, x2, y2), ...]
    """
    # 为相机后方的点创建一个反向投影矩阵
    K_b = build_projection_matrix(K[0,2]*2, K[1,2]*2, camera_bp.get_attribute("fov").as_float(), is_behind_camera=True)
    
    # 获取边界框的世界坐标顶点
    verts = [v for v in actor.bounding_box.get_world_vertices(actor.get_transform())]
    projection = []
    
    # 投影每条边
    for edge in EDGES:
        p1 = get_image_point(verts[edge[0]], K, world_2_camera)
        p2 = get_image_point(verts[edge[1]], K, world_2_camera)
        p1_in_canvas = point_in_canvas(p1, K[1,2]*2, K[0,2]*2)
        p2_in_canvas = point_in_canvas(p2, K[1,2]*2, K[0,2]*2)
        
        # 如果两点都不在画布内，跳过
        if not p1_in_canvas and not p2_in_canvas:
            continue
        
        # 检查点是否在相机前方
        ray0 = verts[edge[0]] - camera.get_transform().location
        ray1 = verts[edge[1]] - camera.get_transform().location
        cam_forward_vec = camera.get_transform().get_forward_vector()
        
        # 如果点在相机后方，使用反向投影矩阵
        if not (cam_forward_vec.dot(ray0) > 0):
            p1 = get_image_point(verts[edge[0]], K_b, world_2_camera)
        if not (cam_forward_vec.dot(ray1) > 0):
            p2 = get_image_point(verts[edge[1]], K_b, world_2_camera)
            
        projection.append((int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])))
    
    return projection
