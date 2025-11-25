import numpy as np
import pygame

# --- 常量配置 ---
YOLO_TARGET_CLASSES = ['car', 'truck', 'bus', 'motorcycle']
VEHICLE_COLOR = (0, 0, 255)  # 蓝色
EDGES = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5], [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]

def build_projection_matrix(w, h, fov, is_behind_camera=False):
    """构建 3D 到 2D 的投影矩阵"""
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
    """将世界坐标点转换为图像像素点"""
    point = np.array([loc.x, loc.y, loc.z, 1])
    point_camera = np.dot(w2c, point)
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
    point_img = np.dot(K, point_camera)
    if point_img[2] != 0:
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]
    return point_img[0:2]

def point_in_canvas(pos, img_h, img_w):
    """检查点是否在图像范围内"""
    if (pos[0] >= 0) and (pos[0] < img_w) and (pos[1] >= 0) and (pos[1] < img_h):
        return True
    return False

def decode_instance_segmentation(img_rgba: np.ndarray):
    """解码 Carla 的实例分割图像"""
    semantic_labels = img_rgba[..., 2]
    actor_ids = img_rgba[..., 1].astype(np.uint16) + (img_rgba[..., 0].astype(np.uint16) << 8)
    return semantic_labels, actor_ids

def process_depth_image(image):
    """
    将 CARLA 深度图像 (BGRA) 转换为以米为单位的深度图 (2D float32 array)
    """
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array.astype(np.float32)
    # (R + G*256 + B*256*256) / (256^3 - 1) * 1000
    normalized_depth = (array[:, :, 2] + array[:, :, 1] * 256 + array[:, :, 0] * 256 * 256) / (256 * 256 * 256 - 1)
    depth_meters = normalized_depth * 1000.0
    return depth_meters

def draw_carla_image(surface, img):
    """绘制 Carla 图像到 Pygame surface"""
    rgb_img = img[:, :, :3][:, :, ::-1]
    frame_surface = pygame.surfarray.make_surface(np.transpose(rgb_img, (1, 0, 2)))
    surface.blit(frame_surface, (0, 0))
    return rgb_img

def get_truth_3d_box_projection(actor, ego, camera_bp, camera, K, world_2_camera):
    """计算真值 3D 框的投影线段"""
    K_b = build_projection_matrix(K[0, 2] * 2, K[1, 2] * 2, camera_bp.get_attribute("fov").as_float(),
                                  is_behind_camera=True)
    verts = [v for v in actor.bounding_box.get_world_vertices(actor.get_transform())]
    projection = []

    for edge in EDGES:
        p1 = get_image_point(verts[edge[0]], K, world_2_camera)
        p2 = get_image_point(verts[edge[1]], K, world_2_camera)
        p1_in_canvas = point_in_canvas(p1, K[1, 2] * 2, K[0, 2] * 2)
        p2_in_canvas = point_in_canvas(p2, K[1, 2] * 2, K[0, 2] * 2)
        if not p1_in_canvas and not p2_in_canvas:
            continue
        ray0 = verts[edge[0]] - camera.get_transform().location
        ray1 = verts[edge[1]] - camera.get_transform().location
        cam_forward_vec = camera.get_transform().get_forward_vector()
        if not (cam_forward_vec.dot(ray0) > 0):
            p1 = get_image_point(verts[edge[0]], K_b, world_2_camera)
        if not (cam_forward_vec.dot(ray1) > 0):
            p2 = get_image_point(verts[edge[1]], K_b, world_2_camera)
        projection.append((int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])))
    return projection