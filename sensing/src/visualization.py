#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
可视化模块
包含绘制图像、边界框、标签和性能信息的函数
"""

import pygame
import numpy as np
import math

from .config import VEHICLE_COLOR, PEDESTRIAN_COLOR, BICYCLE_COLOR, YOLO_TARGET_CLASSES
from .utils import get_truth_3d_box_projection

def draw_carla_image(surface, img):
    """
    将CARLA图像（BGRA）绘制到Pygame surface上
    
    参数:
        surface: Pygame显示表面
        img: BGRA格式的图像数组
        
    返回:
        RGB格式的图像数组，供YOLO使用
    """
    # 转换BGRA为RGB并翻转颜色通道
    rgb_img = img[:, :, :3][:, :, ::-1]
    
    # 创建Pygame表面并绘制
    frame_surface = pygame.surfarray.make_surface(np.transpose(rgb_img, (1, 0, 2)))
    surface.blit(frame_surface, (0, 0))
    
    return rgb_img

def draw_yolo_and_truth(
        surface, font, yolo_results, yolo_model, 
        depth_map, 
        actor_ids_map, 
        world, ego_vehicle, 
        camera_bp, camera_left, K_left, world_2_camera_left, 
        vehicle_tracker
    ):
    """
    绘制YOLO检测结果和真值对比
    
    1. 使用YOLO找到2D框和Track ID
    2. 从深度图中测量距离(Z)
    3. 将(Track ID, Z)喂给卡尔曼滤波器
    4. 从滤波器中获取平滑后的距离和速度
    5. 获取真值3D框和真值速度（用于对比）
    6. 绘制真值3D框，并标注算法速度vs真值速度
    
    参数:
        surface: Pygame显示表面
        font: Pygame字体对象
        yolo_results: YOLO检测结果
        yolo_model: YOLO模型实例
        depth_map: 深度图像数组
        actor_ids_map: 实例分割的Actor ID映射
        world: CARLA世界对象
        ego_vehicle: 主角车辆
        camera_bp: 相机蓝图
        camera_left: 相机Actor
        K_left: 相机内参矩阵
        world_2_camera_left: 世界到相机的变换矩阵
        vehicle_tracker: 车辆跟踪器实例
    """
    drawn_actor_ids = set()

    # 处理每个YOLO结果
    for r in yolo_results:
        # 跳过没有ID的结果
        if r.boxes.id is None:
            continue

        # 获取检测框、跟踪ID、类别和置信度
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        track_ids = r.boxes.id.cpu().numpy().astype(int)
        class_ids = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy().astype(float)

        # 处理每个检测框
        for box, track_id, class_id, conf in zip(boxes, track_ids, class_ids, confs):
            # 提取边界框坐标
            x1, y1, x2, y2 = box
            class_name = yolo_model.names.get(class_id)
            
            # 只处理目标车辆类别
            if class_name not in YOLO_TARGET_CLASSES:
                continue
            
            # 从深度图中获取距离
            roi = depth_map[y1:y2, x1:x2]
            z_values = roi[roi > 0]  # 过滤掉无效深度值

            if len(z_values) == 0:
                continue
                
            # 使用中位数获取稳定的距离测量
            measured_Z = np.median(z_values)

            # 更新卡尔曼滤波器并获取平滑后的距离和速度
            kf = vehicle_tracker.get_filter(track_id, measured_Z)
            kf.predict()
            kf.update(np.array([measured_Z]))
            
            computed_Z_smooth = kf.x[0]
            computed_vel_mps = kf.x[1]
            
            # 计算主角车速度，用于转换相对速度为绝对速度
            ego_vel_vec = ego_vehicle.get_velocity()
            ego_speed_mps = math.sqrt(ego_vel_vec.x**2 + ego_vel_vec.y**2 + ego_vel_vec.z**2)
            
            # 计算绝对速度（考虑相对运动）
            computed_abs_vel_mps = ego_speed_mps + computed_vel_mps
            computed_vel_kmh = computed_abs_vel_mps * 3.6

            # 获取真值信息
            cx = (x1 + x2) // 2  # 边界框中心点
            cy = (y1 + y2) // 2
            
            # 检查中心点是否在有效范围内
            if not (0 <= cy < actor_ids_map.shape[0] and 0 <= cx < actor_ids_map.shape[1]):
                continue
            
            # 获取对应位置的Actor ID
            actor_id = int(actor_ids_map[cy, cx])
            
            # 跳过无效ID、主角车或已绘制的车辆
            if actor_id == 0 or actor_id == ego_vehicle.id or actor_id in drawn_actor_ids:
                continue
            
            # 获取CARLA Actor对象
            actor = world.get_actor(actor_id)
            
            # 验证Actor是否存在且为车辆、行人或自行车
            if actor is None or not actor.is_alive or not (isinstance(actor, carla.Vehicle) or isinstance(actor, carla.Walker)):
                continue
                
            drawn_actor_ids.add(actor_id)

            # 获取真值速度
            vel = actor.get_velocity()
            truth_vel_kmh = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            
            # 获取真值3D边界框投影
            projection = get_truth_3d_box_projection(
                actor, ego_vehicle, camera_bp, 
                camera_left, K_left, world_2_camera_left
            )

            # 根据对象类型选择不同颜色
            if isinstance(actor, carla.Walker):
                draw_color = PEDESTRIAN_COLOR
            elif class_name == 'bicycle' or class_name == 'motorcycle':
                draw_color = BICYCLE_COLOR
            else:
                draw_color = VEHICLE_COLOR

            # 绘制3D边界框
            n = 0
            mean_x = 0
            mean_y = 0
            for line in projection:
                pygame.draw.line(surface, draw_color, (line[0], line[1]), (line[2], line[3]), 2)
                mean_x += line[0]
                mean_y += line[1]
                n += 1

            # 绘制标签（算法速度vs真值速度）
            if n > 0:
                mean_x /= n
                mean_y /= n
                
                # 距离标签
                label_dist = f"Dist (Alg): {computed_Z_smooth:.1f} m"
                text_surface_dist = font.render(label_dist, True, (255, 255, 255), draw_color)
                text_rect_dist = text_surface_dist.get_rect(bottomleft=(mean_x, mean_y - 5))
                surface.blit(text_surface_dist, text_rect_dist)
                
                # 速度标签
                label_vel = f"Vel (Alg): {computed_vel_kmh:.1f} km/h | Truth: {truth_vel_kmh:.1f} km/h"
                text_surface_vel = font.render(label_vel, True, (255, 255, 255), draw_color)
                text_rect_vel = text_surface_vel.get_rect(bottomleft=(mean_x, text_rect_dist.top - 5))
                surface.blit(text_surface_vel, text_rect_vel)

def draw_fps_info(surface, font, inference_time):
    """
    在屏幕上绘制模型推理时间信息
    
    参数:
        surface: Pygame显示表面
        font: Pygame字体对象
        inference_time: 推理时间（毫秒）
    """
    time_text = f"YOLO Inference: {inference_time:.1f} ms"
    time_surface = font.render(time_text, True, (255, 255, 0))  # 黄色文字
    surface.blit(time_surface, (10, 10))

# 导入carla以避免循环导入问题
import carla
