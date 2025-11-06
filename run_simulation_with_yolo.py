#!/usr/bin/env python

import carla
import queue
import pygame
import argparse
import numpy as np
import logging
import sys
import os
import time
import random
import math

# --- (删除) 不再需要 OpenCV ---
# try:
#     import cv2
# except ImportError:
#     ...

# --- (保留) 卡尔曼滤波器库 ---
try:
    from filterpy.kalman import KalmanFilter
    from filterpy.common import Q_discrete_white_noise
except ImportError:
    raise RuntimeError('需要 filterpy, 请运行 "pip install filterpy"')

# --- YOLOv11 Import ---
try:
    from ultralytics import YOLO
except ImportError:
    raise RuntimeError('cannot import ultralytics, please run "pip install ultralytics"')

from pygame.locals import K_ESCAPE

# --- 我们的目标现在是车辆 ---
YOLO_TARGET_CLASSES = ['car', 'truck', 'bus', 'motorcycle']
VEHICLE_COLOR = (0, 0, 255) # 蓝色
EDGES = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]


# ==============================================================================
# -- 3D->2D 投影辅助函数 (来自 bounding_boxes.py) -----------------------------
# ==============================================================================
#
def build_projection_matrix(w, h, fov, is_behind_camera=False):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    if is_behind_camera:
        K[0, 0] = K[1, 1] = -focal
    else:
        K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

#
def get_image_point(loc, K, w2c):
    point = np.array([loc.x, loc.y, loc.z, 1])
    point_camera = np.dot(w2c, point)
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
    point_img = np.dot(K, point_camera)
    if point_img[2] != 0:
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]
    return point_img[0:2]

#
def point_in_canvas(pos, img_h, img_w):
    if (pos[0] >= 0) and (pos[0] < img_w) and (pos[1] >= 0) and (pos[1] < img_h):
        return True
    return False

#
def decode_instance_segmentation(img_rgba: np.ndarray):
    semantic_labels = img_rgba[..., 2]
    actor_ids = img_rgba[..., 1].astype(np.uint16) + (img_rgba[..., 0].astype(np.uint16) << 8)
    return semantic_labels, actor_ids

#
def get_truth_3d_box_projection(actor, ego, camera_bp, camera, K, world_2_camera):
    """
    (已重命名) 从 CARLA 真值获取 3D 边界框的 2D 投影。
    """
    K_b = build_projection_matrix(K[0,2]*2, K[1,2]*2, camera_bp.get_attribute("fov").as_float(), is_behind_camera=True)
    verts = [v for v in actor.bounding_box.get_world_vertices(actor.get_transform())]
    projection = []
    
    for edge in EDGES:
        p1 = get_image_point(verts[edge[0]], K, world_2_camera)
        p2 = get_image_point(verts[edge[1]],  K, world_2_camera)
        p1_in_canvas = point_in_canvas(p1, K[1,2]*2, K[0,2]*2)
        p2_in_canvas = point_in_canvas(p2, K[1,2]*2, K[0,2]*2)
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

# ==============================================================================
# -- (新增) 深度图处理函数 ----------------------------------------------------
# ==============================================================================

def process_depth_image(image):
    """
    将 CARLA 深度图像 (BGRA) 转换为一个 2D 数组，
    其中每个像素的值是其以米为单位的深度。
    """
    # CARLA 深度图编码在 [R, G, B] 通道中
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array.astype(np.float32) # 转换为浮点数
    
    # 将 [R, G, B] 通道合并为深度
    # (B * 256 * 256 + G * 256 + R) / (256 * 256 * 256 - 1)
    # (类似逻辑)
    normalized_depth = (array[:, :, 2] + array[:, :, 1] * 256 + array[:, :, 0] * 256 * 256) / (256 * 256 * 256 - 1)
    
    # 乘以 1000.0 (摄像机远裁剪平面) 得到以米为单位的深度
    depth_meters = normalized_depth * 1000.0
    return depth_meters

# ==============================================================================
# -- (保留) 卡尔曼滤波器跟踪器 ------------------------------------------------
# ==============================================================================

class VehicleTracker(object):
    """
    为每个 track_id 管理一个独立的卡尔曼滤波器，
    以平滑 *深度相机* 的距离并估算速度。
    """
    def __init__(self, dt):
        self.dt = dt # 时间步长 (0.05s)
        self.filters = {} # 存储 {track_id: KalmanFilter}

    def _create_filter(self, initial_z):
        """
        创建一个新的卡尔曼滤波器
        状态 x = [z, v_z] (距离, 速度)
        测量 z = [z]       (只有距离)
        """
        kf = KalmanFilter(dim_x=2, dim_z=1)
        
        # 1. 初始状态: [测量的距离, 0 速度]
        kf.x = np.array([initial_z, 0.0])
        
        # 2. 状态转移矩阵 F (过程模型)
        kf.F = np.array([[1., self.dt],
                         [0., 1.]])
        
        # 3. 测量函数 H (我们只能测量距离)
        kf.H = np.array([[1., 0.]])
        
        # 4. 协方差 P (我们对初始状态的不确定性)
        kf.P *= 10.0 # 不太确定
        
        # 5. 测量噪声 R (关键!)
        # --- (已修改) ---
        # 深度相机非常准确，所以我们*非常相信*测量
        kf.R = np.array([[0.01]]) # <--- 调低这个值 = "更相信测量"
        
        # 6. 过程噪声 Q (我们相信物理模型)
        kf.Q = Q_discrete_white_noise(dim=2, dt=self.dt, var=0.1)
        
        return kf

    def get_filter(self, track_id, computed_Z):
        """
        获取或创建此 track_id 的滤波器
        """
        if track_id not in self.filters:
            # 如果是新目标, 创建一个滤波器
            self.filters[track_id] = self._create_filter(computed_Z)
        return self.filters[track_id]


# ==============================================================================
# -- Visualization Functions ---------------------------------------------------
# ==============================================================================

def draw_carla_image(surface, img):
    """
    将 CARLA 图像（BGRA）绘制到 Pygame surface 背景上。
    同时返回转换后的 RGB 图像供 YOLO 使用。
   
    """
    rgb_img = img[:, :, :3][:, :, ::-1]
    frame_surface = pygame.surfarray.make_surface(np.transpose(rgb_img, (1, 0, 2)))
    surface.blit(frame_surface, (0, 0))
    return rgb_img

def draw_yolo_and_truth(
        surface, font, yolo_results, yolo_model, 
        depth_map, # <--- (已修改) 算法深度图
        actor_ids_map, # <--- 真值 ID 查找表
        world, ego_vehicle, 
        camera_bp, camera_left, K_left, world_2_camera_left, 
        vehicle_tracker # <--- 卡尔曼跟踪器
    ):
    """
    (重大修改)
    1. 使用 YOLO 找到 2D 框和 Track ID。
    2. 从 深度图 (depth_map) 中 *测量* 距离 (Z)。
    3. 将 (Track ID, Z) 喂给 *卡尔曼滤波器*。
    4. 从滤波器中获取 *平滑后的* 距离和速度。
    5. 获取 *真值* 3D 框和 *真值* 速度 (用于对比)。
    6. 绘制 *真值* 3D 框，并标注 *算法速度* vs *真值速度*。
   
    """
    
    drawn_actor_ids = set()

    for r in yolo_results:
        if r.boxes.id is None:
            continue

        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        track_ids = r.boxes.id.cpu().numpy().astype(int)
        class_ids = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy().astype(float)

        for box, track_id, class_id, conf in zip(boxes, track_ids, class_ids, confs):
            
            x1, y1, x2, y2 = box
            class_name = yolo_model.names.get(class_id)
            
            if class_name not in YOLO_TARGET_CLASSES:
                continue
            
            # --- 2. 算法：测量距离 (Z) ---
            # (已修改) 从深度图中获取距离
            roi = depth_map[y1:y2, x1:x2]
            z_values = roi[roi > 0] # 过滤掉 0
            
            if len(z_values) == 0:
                continue
                
            measured_Z = np.median(z_values) # 稳定、准确的测量值

            # --- 3. 算法：卡尔曼滤波器更新 ---
            kf = vehicle_tracker.get_filter(track_id, measured_Z)
            kf.predict()
            kf.update(np.array([measured_Z]))
            
            computed_Z_smooth = kf.x[0]
            computed_vel_mps = kf.x[1]
            # (已修改) 我们估算的是相对速度，真值是绝对速度。
            # 为了公平对比，我们计算 *主角车* 的速度，并将其加到我们的相对速度上。
            ego_vel_vec = ego_vehicle.get_velocity()
            ego_speed_mps = math.sqrt(ego_vel_vec.x**2 + ego_vel_vec.y**2 + ego_vel_vec.z**2)
            
            # v_relative = v_truth - v_ego
            # v_truth = v_relative + v_ego
            # 我们的 v_z (computed_vel_mps) 是 v_relative (因为深度在减小)
            # 注意: 我们的 v_z 是负的 (接近)，所以我们用 ego_speed - v_z
            computed_abs_vel_mps = ego_speed_mps + computed_vel_mps
            computed_vel_kmh = computed_abs_vel_mps * 3.6


            # --- 4. 真值：获取 Actor ID 和速度 (用于对比) ---
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            if not (0 <= cy < actor_ids_map.shape[0] and 0 <= cx < actor_ids_map.shape[1]):
                continue
            
            actor_id = int(actor_ids_map[cy, cx])
            
            if actor_id == 0 or actor_id == ego_vehicle.id or actor_id in drawn_actor_ids:
                continue
            
            actor = world.get_actor(actor_id)
            
            if actor is None or not actor.is_alive or not isinstance(actor, carla.Vehicle):
                continue
                
            drawn_actor_ids.add(actor_id)

            # 获取真值速度
            vel = actor.get_velocity()
            truth_vel_kmh = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            
            # --- 5. 真值：获取 3D 边界框 (用于绘制) ---
            projection = get_truth_3d_box_projection(
                actor, ego_vehicle, camera_bp, 
                camera_left, K_left, world_2_camera_left
            )

            # --- 6. 绘制 3D 边界框 (真值) ---
            n = 0
            mean_x = 0
            mean_y = 0
            for line in projection:
                pygame.draw.line(surface, VEHICLE_COLOR, (line[0], line[1]), (line[2],line[3]), 2)
                mean_x += line[0]
                mean_y += line[1]
                n += 1

            # --- 7. 绘制标签 (算法速度 vs 真值速度) ---
            if n > 0:
                mean_x /= n
                mean_y /= n
                
                label_dist = f"Dist (Alg): {computed_Z_smooth:.1f} m"
                label_vel = f"Vel (Alg): {computed_vel_kmh:.1f} km/h | Truth: {truth_vel_kmh:.1f} km/h"
                
                text_surface_dist = font.render(label_dist, True, (255, 255, 255), VEHICLE_COLOR)
                text_rect_dist = text_surface_dist.get_rect(bottomleft=(mean_x, mean_y - 5))
                surface.blit(text_surface_dist, text_rect_dist)
                
                text_surface_vel = font.render(label_vel, True, (255, 255, 255), VEHICLE_COLOR)
                text_rect_vel = text_surface_vel.get_rect(bottomleft=(mean_x, text_rect_dist.top - 5))
                surface.blit(text_surface_vel, text_rect_vel)


# ==============================================================================
# -- CarlaYoloRunner Class -----------------------------------------------------
# ==============================================================================

class CarlaYoloRunner(object):
    """主运行类，封装了所有状态和逻辑"""
    
    def __init__(self, args):
        self.args = args
        self.width = args.width
        self.height = args.height
        self.run_simulation = True
        self.display = None
        self.clock = None
        self.yolo_font = None
        self.client = None
        self.world = None
        self.ego_vehicle = None
        
        # --- (已修改) RGB + 深度 + 分割 ---
        self.camera_rgb = None
        self.camera_depth = None
        self.inst_camera = None
        
        self.camera_bp_rgb = None # <--- (已修改)
        
        self.rgb_queue = queue.Queue()
        self.depth_queue = queue.Queue()
        self.inst_queue = queue.Queue()
        
        self.traffic_manager = None
        self.actors_to_destroy = []
        self.walker_controllers = []
        
        self.model = None
        self.last_inference_time = 0.0

        # --- (已删除) SGBM 和 Q 矩阵 ---
        # self.stereo = None
        # self.Q = None
        
        # --- (保留) 卡尔曼滤波器跟踪器 ---
        self.vehicle_tracker = VehicleTracker(dt=0.05) # dt=0.05s (20 FPS)
        
    def _init_pygame(self):
        """初始化 Pygame"""
        logging.info("初始化 Pygame...")
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(
            (self.width, self.height), 
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        self.display.fill((0, 0, 0))
        pygame.display.flip()
        pygame.display.set_caption("CARLA (YOLO+深度) 3D 速度检测 (20 FPS)")
        self.clock = pygame.time.Clock()
        self.yolo_font = pygame.font.SysFont("Arial", 16)

    def _load_yolo_model(self):
        """加载 YOLOv11 模型"""
        logging.info(f"正在从 {self.args.yolo_model} 加载 YOLO11 模型...")
        try:
            self.model = YOLO(self.args.yolo_model)
            logging.info(f"YOLO11 模型加载成功。")
            logging.info(f"将跟踪以下类别: {YOLO_TARGET_CLASSES}")
        except Exception as e:
            logging.error(f"加载 YOLO 模型失败: {e}")
            raise

    def _setup_carla(self):
        """连接 CARLA 并设置世界和交通管理器"""
        logging.info("正在连接 CARLA...")
        self.client = carla.Client(self.args.host, self.args.port)
        self.client.set_timeout(10.0)

        logging.info("正在加载地图 Town05...")
        self.world = self.client.load_world('Town05')

        tm_port = self.args.tm_port
        logging.info(f"正在连接到交通管理器 (端口: {tm_port})...")
        self.traffic_manager = self.client.get_trafficmanager(tm_port)
        self.traffic_manager.set_synchronous_mode(True)

        # 确保 20 FPS
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        logging.info("CARLA 同步模式已启用 (20 FPS)。")
        
        self.world.tick()

    def _spawn_actors(self):
        """(已修改) 生成主角、NPC 和 *三个* 摄像头"""
        logging.info("正在生成 Actors...")
        bp_lib = self.world.get_blueprint_library()
        
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            logging.error("地图 'Town05' 没有可用的生成点。")
            raise RuntimeError("没有可用的生成点")

        # --- 1. 生成主角车辆 (Ego Vehicle) ---
        vehicle_bp = bp_lib.find('vehicle.tesla.cybertruck')
        vehicle_bp.set_attribute('role_name', 'ego')
        transform = random.choice(spawn_points)
        spawn_points.remove(transform)
        
        self.ego_vehicle = self.world.try_spawn_actor(vehicle_bp, transform)
        if self.ego_vehicle is None:
            logging.error("无法生成主角车辆。")
            raise RuntimeError("无法生成主角车辆")

        self.ego_vehicle.set_autopilot(True, self.traffic_manager.get_port())
        self.traffic_manager.vehicle_percentage_speed_difference(self.ego_vehicle, -30)
        self.actors_to_destroy.append(self.ego_vehicle)
        logging.info(f"已生成主角车辆 (ID: {self.ego_vehicle.id})")

        # --- 2. 生成传感器 ---
        
        # (共享 FOV 和位置)
        fov = 70.0
        camera_init_trans = carla.Transform(carla.Location(x=1.2, z=2.0))
        
        # (RGB 相机)
        self.camera_bp_rgb = bp_lib.find('sensor.camera.rgb')
        self.camera_bp_rgb.set_attribute('image_size_x', str(self.width))
        self.camera_bp_rgb.set_attribute('image_size_y', str(self.height))
        self.camera_bp_rgb.set_attribute('fov', str(fov))
        
        self.camera_rgb = self.world.spawn_actor(
            self.camera_bp_rgb, camera_init_trans, attach_to=self.ego_vehicle
        )
        self.actors_to_destroy.append(self.camera_rgb)
        self.camera_rgb.listen(self.rgb_queue.put)
        logging.info(f"已生成 RGB 相机")

        # --- (新增) 深度相机 (用于算法) ---
        camera_bp_depth = bp_lib.find('sensor.camera.depth')
        camera_bp_depth.set_attribute('image_size_x', str(self.width))
        camera_bp_depth.set_attribute('image_size_y', str(self.height))
        camera_bp_depth.set_attribute('fov', str(fov))
        
        self.camera_depth = self.world.spawn_actor(
            camera_bp_depth, camera_init_trans, attach_to=self.ego_vehicle
        )
        self.actors_to_destroy.append(self.camera_depth)
        self.camera_depth.listen(self.depth_queue.put)
        logging.info(f"已生成深度相机")
        
        # (实例分割相机 - 用于获取真值 ID)
        inst_camera_bp = bp_lib.find('sensor.camera.instance_segmentation')
        inst_camera_bp.set_attribute('image_size_x', str(self.width))
        inst_camera_bp.set_attribute('image_size_y', str(self.height))
        inst_camera_bp.set_attribute('fov', str(fov))
        
        self.inst_camera = self.world.spawn_actor(
            inst_camera_bp, camera_init_trans, attach_to=self.ego_vehicle
        )
        self.actors_to_destroy.append(self.inst_camera)
        self.inst_camera.listen(self.inst_queue.put)
        logging.info("已生成实例分割相机 (用于真值对比)")
        
        # --- 3. 生成 NPC 车辆 ---
        logging.info(f"正在生成 {self.args.num_vehicles} 辆 NPC 车辆...")
        vehicle_blueprints = bp_lib.filter('vehicle.*')
        
        for _ in range(self.args.num_vehicles):
            if not spawn_points:
                logging.warning("没有更多可用的生成点给NPC车辆了。")
                break
            bp = random.choice(vehicle_blueprints)
            if bp.id == vehicle_bp.id: continue
            transform = random.choice(spawn_points)
            spawn_points.remove(transform)
            npc_vehicle = self.world.try_spawn_actor(bp, transform)
            if npc_vehicle:
                self.actors_to_destroy.append(npc_vehicle)
                npc_vehicle.set_autopilot(True, self.traffic_manager.get_port())

        # --- 4. 生成 NPC 行人 ---
        logging.info(f"正在生成 {self.args.num_walkers} 个 NPC 行人...")
        walker_blueprints = bp_lib.filter('walker.pedestrian.*')
        
        for _ in range(self.args.num_walkers):
            spawn_location = self.world.get_random_location_from_navigation()
            if not spawn_location:
                logging.warning("无法找到行人导航点。")
                break
            bp = random.choice(walker_blueprints)
            transform = carla.Transform(spawn_location)
            walker = self.world.try_spawn_actor(bp, transform)
            if walker:
                self.actors_to_destroy.append(walker)
                controller_bp = bp_lib.find('controller.ai.walker')
                controller = self.world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
                if controller:
                    self.walker_controllers.append(controller)
                    self.actors_to_destroy.append(controller)
                    controller.start()
                    controller.go_to_location(self.world.get_random_location_from_navigation())
                    controller.set_max_speed(1 + random.random())
                else:
                    logging.warning("无法生成行人控制器。")
                    walker.destroy()
        logging.info("Actors 生成完毕。")

    def _process_events(self):
        """处理 Pygame 事件 (退出)"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.run_simulation = False
            elif event.type == pygame.KEYUP:
                if event.key == K_ESCAPE:
                    self.run_simulation = False

    def _game_loop(self):
        """(已修改) 核心游戏循环"""
        logging.info("开始 YOLOv11 车辆检测...")
        while self.run_simulation:
            # 1. 处理退出事件
            self._process_events()

            # 2. CARLA 世界步进
            self.world.tick()

            # 3. 获取传感器数据
            try:
                image_rgb = self.rgb_queue.get(timeout=1.0)
                image_depth = self.depth_queue.get(timeout=1.0)
                inst_seg_image = self.inst_queue.get(timeout=1.0)
            except queue.Empty:
                logging.warning("传感器队列超时。")
                continue
                
            # --- 4. 图像预处理 ---
            img_bgra_rgb = np.reshape(np.copy(image_rgb.raw_data), (self.height, self.width, 4))
            img_rgb = draw_carla_image(self.display, img_bgra_rgb)
            
            # (新增) 处理深度图
            depth_map = process_depth_image(image_depth)

            # (保留) 处理分割图
            inst_seg_bgra = np.reshape(np.copy(inst_seg_image.raw_data), (self.height, self.width, 4))
            semantic_labels, actor_ids_map = decode_instance_segmentation(inst_seg_bgra)
            
            # --- 5. (已删除) StereoSGBM ---

            # --- 6. 算法：运行 YOLO 跟踪 ---
            yolo_results = self.model.track(img_rgb, persist=True, verbose=False)
            
            if yolo_results:
                self.last_inference_time = yolo_results[0].speed.get('inference', 0.0)
            
            # --- 7. 获取真值投影矩阵 ---
            world_2_camera = np.array(self.camera_rgb.get_transform().get_inverse_matrix())
            K = build_projection_matrix(self.width, self.height, self.camera_bp_rgb.get_attribute("fov").as_float())
            
            # --- 8. 融合、计算 和 绘制 ---
            draw_yolo_and_truth(
                self.display, self.yolo_font, yolo_results, self.model,
                depth_map, actor_ids_map, self.world, self.ego_vehicle, 
                self.camera_bp_rgb, self.camera_rgb, K, world_2_camera,
                self.vehicle_tracker
            )

            # --- 9. 绘制模型帧率 ---
            time_text = f"YOLO Inference: {self.last_inference_time:.1f} ms"
            time_surface = self.yolo_font.render(time_text, True, (255, 255, 0)) # 黄色
            self.display.blit(time_surface, (10, 10))
            
            # 10. 更新显示
            pygame.display.flip()
            self.clock.tick(20) # 匹配 20 FPS

    def run(self):
        """按顺序执行所有操作"""
        try:
            self._init_pygame()
            self._load_yolo_model()
            self._setup_carla()
            self._spawn_actors()
            self.world.tick()
            self._game_loop()

        except KeyboardInterrupt:
            logging.info("用户中断...正在退出。")
        except Exception as e:
            logging.error(f"发生未捕获的错误: {e}", exc_info=True)
        finally:
            self.cleanup()

    def cleanup(self):
        """清理所有资源"""
        logging.info("正在清理资源...")

        original_settings = None
        if self.world:
            try:
                original_settings = self.world.get_settings()
            except Exception as e:
                logging.error(f"获取世界设置失败: {e}")

        try:
            if self.world and original_settings:
                original_settings.synchronous_mode = False
                original_settings.fixed_delta_seconds = None
                self.world.apply_settings(original_settings)
                logging.info("CARLA 异步模式已恢复。")
            if self.traffic_manager:
                self.traffic_manager.set_synchronous_mode(False)
        except Exception as e:
            logging.warning(f"恢复异步模式失败: {e}")

        logging.info("正在停止行人AI...")
        for controller in self.walker_controllers:
            if controller and controller.is_alive:
                try:
                    controller.stop()
                except Exception as e:
                    logging.warning(f"停止行人控制器 {controller.id} 失败: {e}")

        logging.info(f"正在销毁 {len(self.actors_to_destroy)} 个 actors...")
        if self.client:
            for actor in self.actors_to_destroy:
                if actor and actor.is_alive and hasattr(actor, 'is_listening') and actor.is_listening:
                    actor.stop()
            
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actors_to_destroy if x and x.is_alive])
        
        self.actors_to_destroy.clear()
        self.walker_controllers.clear()

        pygame.quit()
        logging.info('完成。')


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================

def main():
    # --- 参数解析 ---
    argparser = argparse.ArgumentParser(description='CARLA (YOLO+深度) 3D 速度检测')
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='主机IP (default: 127.0.0.1)')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP端口 (default: 2000)')
    argparser.add_argument('--res', metavar='WIDTHxHEIGHT', default='1280x720',
                           help='窗口分辨率')
    argparser.add_argument('--yolo-model', metavar='PATH', required=True,
                           help='YOLO 模型文件路径 (e.g., yolov11n.pt)')

    argparser.add_argument('--tm-port', metavar='P', default=8000, type=int,
                           help='交通管理器(TM)端口 (default: 8000)')
    argparser.add_argument('-n', '--num-vehicles', metavar='N', default=30, type=int,
                           help='生成的NPC车辆数量 (default: 30)')
    argparser.add_argument('-w', '--num-walkers', metavar='W', default=15, type=int,
                           help='生成的NPC行人数量 (default: 15)')
    
    args = argparser.parse_args()
        
    args.width, args.height = [int(x) for x in args.res.split('x')]
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    # --- 运行 ---
    runner = None
    try:
        runner = CarlaYoloRunner(args)
        runner.run()
    except Exception as e:
        logging.critical(f"程序启动失败: {e}")
    

if __name__ == '__main__':
    print('CARLA (YOLO+深度) 3D 速度检测 说明:')
    print('ESC  : 退出')
    print('正在启动...')
    main()