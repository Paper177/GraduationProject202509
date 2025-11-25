#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
主运行器模块
包含CarlaYoloRunner类，负责整个程序的核心运行逻辑和资源管理
"""

import carla
import queue
import pygame
import logging
import random
import numpy as np
import time

# 导入YOLO库
try:
    from ultralytics import YOLO
except ImportError:
    raise RuntimeError('cannot import ultralytics, please run "pip install ultralytics"')

# 导入我们的模块
from .config import DEFAULT_DT, TARGET_FPS, DEFAULT_CAMERA_LOCATION
from .utils import build_projection_matrix, process_depth_image, decode_instance_segmentation
from .tracker import VehicleTracker
from .visualization import draw_carla_image, draw_yolo_and_truth, draw_fps_info

# 导入Pygame常量
from pygame.locals import K_ESCAPE

class CarlaYoloRunner(object):
    """
    主运行类，封装了所有CARLA模拟器、YOLO检测和可视化的状态和逻辑
    """
    
    def __init__(self, yolo_model_path, conf_threshold=0.25, iou_threshold=0.45,
                 carla_host='localhost', carla_port=2000, world_name='Town03',
                 disable_rendering=False, record_data=False, output_directory='./output',
                 width=800, height=600, tm_port=8000, num_vehicles=50, num_walkers=30):
        """
        初始化运行器
        
        参数:
            yolo_model_path: YOLO模型路径
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值
            carla_host: CARLA服务器主机名
            carla_port: CARLA服务器端口
            world_name: 要加载的CARLA世界名称
            disable_rendering: 是否禁用渲染
            record_data: 是否记录数据
            output_directory: 输出目录
            width: 显示宽度
            height: 显示高度
            tm_port: 交通管理器端口
            num_vehicles: NPC车辆数量
            num_walkers: NPC行人数量
        """
        # 基本配置
        self.width = width
        self.height = height
        self.run_simulation = True
        # 保存参数供后续使用
        self.yolo_model_path = yolo_model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.carla_host = carla_host
        self.carla_port = carla_port
        self.world_name = world_name
        self.disable_rendering = disable_rendering
        self.record_data = record_data
        self.output_directory = output_directory
        self.tm_port = tm_port
        self.num_vehicles = num_vehicles
        self.num_walkers = num_walkers
        
        # Pygame相关
        self.display = None
        self.clock = None
        self.yolo_font = None
        
        # CARLA相关
        self.client = None
        self.world = None
        self.ego_vehicle = None
        
        # 相机传感器
        self.camera_rgb = None
        self.camera_depth = None
        self.inst_camera = None
        self.camera_bp_rgb = None
        
        # 传感器数据队列
        self.rgb_queue = queue.Queue()
        self.depth_queue = queue.Queue()
        self.inst_queue = queue.Queue()
        
        # CARLA交通管理
        self.traffic_manager = None
        self.actors_to_destroy = []
        self.walker_controllers = []
        
        # YOLO模型
        self.model = None
        self.last_inference_time = 0.0
        
        # 车辆跟踪器
        self.vehicle_tracker = VehicleTracker(dt=DEFAULT_DT)
    
    def _init_pygame(self):
        """
        初始化Pygame显示环境
        """
        logging.info("初始化 Pygame...")
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(
            (self.width, self.height), 
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        self.display.fill((0, 0, 0))
        pygame.display.flip()
        pygame.display.set_caption("CARLA (YOLO+深度) 3D 速度检测")
        self.clock = pygame.time.Clock()
        self.yolo_font = pygame.font.SysFont("Arial", 16)

    def _load_yolo_model(self):
        """
        加载YOLOv11模型
        """
        logging.info(f"正在从 {self.yolo_model_path} 加载 YOLO11 模型...")
        try:
            self.model = YOLO(self.yolo_model_path)
            logging.info(f"YOLO11 模型加载成功。")
        except Exception as e:
            logging.error(f"加载 YOLO 模型失败: {e}")
            raise

    def _setup_carla(self):
        """
        连接CARLA并设置世界和交通管理器
        """
        logging.info("正在连接 CARLA...")
        self.client = carla.Client(self.carla_host, self.carla_port)
        self.client.set_timeout(10.0)

        logging.info(f"正在加载地图 {self.world_name}...")
        self.world = self.client.load_world(self.world_name)

        logging.info(f"正在连接到交通管理器 (端口: {self.tm_port})...")
        self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
        self.traffic_manager.set_synchronous_mode(True)

        # 确保指定的帧率
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / TARGET_FPS
        self.world.apply_settings(settings)
        logging.info(f"CARLA 同步模式已启用 ({TARGET_FPS} FPS)。")
        
        self.world.tick()

    def _spawn_actors(self):
        """
        生成主角、NPC和传感器
        """
        logging.info("正在生成 Actors...")
        bp_lib = self.world.get_blueprint_library()
        
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            logging.error(f"地图 '{self.world_name}' 没有可用的生成点。")
            raise RuntimeError("没有可用的生成点")

        # 1. 生成主角车辆
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

        # 2. 生成传感器
        
        # 共享相机参数
        fov = 70.0
        camera_init_trans = carla.Transform(
            carla.Location(x=DEFAULT_CAMERA_LOCATION[0], 
                         y=DEFAULT_CAMERA_LOCATION[1], 
                         z=DEFAULT_CAMERA_LOCATION[2])
        )
        
        # RGB相机
        self.camera_bp_rgb = bp_lib.find('sensor.camera.rgb')
        self.camera_bp_rgb.set_attribute('image_size_x', str(self.width))
        self.camera_bp_rgb.set_attribute('image_size_y', str(self.height))
        self.camera_bp_rgb.set_attribute('fov', str(fov))
        
        self.camera_rgb = self.world.spawn_actor(
            self.camera_bp_rgb, camera_init_trans, attach_to=self.ego_vehicle
        )
        self.actors_to_destroy.append(self.camera_rgb)
        self.camera_rgb.listen(self.rgb_queue.put)
        logging.info("已生成 RGB 相机")

        # 深度相机
        camera_bp_depth = bp_lib.find('sensor.camera.depth')
        camera_bp_depth.set_attribute('image_size_x', str(self.width))
        camera_bp_depth.set_attribute('image_size_y', str(self.height))
        camera_bp_depth.set_attribute('fov', str(fov))
        
        self.camera_depth = self.world.spawn_actor(
            camera_bp_depth, camera_init_trans, attach_to=self.ego_vehicle
        )
        self.actors_to_destroy.append(self.camera_depth)
        self.camera_depth.listen(self.depth_queue.put)
        logging.info("已生成深度相机")
        
        # 实例分割相机
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
        
        # 3. 生成NPC车辆
        logging.info(f"正在生成 {self.num_vehicles} 辆 NPC 车辆...")
        vehicle_blueprints = bp_lib.filter('vehicle.*')
        
        for _ in range(self.num_vehicles):
            if not spawn_points:
                logging.warning("没有更多可用的生成点给NPC车辆了。")
                break
            bp = random.choice(vehicle_blueprints)
            if bp.id == vehicle_bp.id: 
                continue
            transform = random.choice(spawn_points)
            spawn_points.remove(transform)
            npc_vehicle = self.world.try_spawn_actor(bp, transform)
            if npc_vehicle:
                self.actors_to_destroy.append(npc_vehicle)
                npc_vehicle.set_autopilot(True, self.traffic_manager.get_port())

        # 4. 生成NPC行人
        logging.info(f"正在生成 {self.num_walkers} 个 NPC 行人...")
        walker_blueprints = bp_lib.filter('walker.pedestrian.*')
        
        for _ in range(self.num_walkers):
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
        """
        处理Pygame事件（主要是退出事件）
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.run_simulation = False
            elif event.type == pygame.KEYUP:
                if event.key == K_ESCAPE:
                    self.run_simulation = False

    def _game_loop(self, duration=None):
        """
        核心游戏循环，处理传感器数据、运行YOLO检测和可视化
        
        参数:
            duration: 模拟持续时间（秒），如果为None则持续运行直到用户中断
        """
        logging.info("开始 YOLOv11 车辆检测...")
        start_time = time.time()
        
        while self.run_simulation:
            # 检查是否达到指定的模拟持续时间
            if duration is not None and (time.time() - start_time) >= duration:
                logging.info(f"已达到指定的模拟持续时间 ({duration} 秒)，准备退出...")
                self.run_simulation = False
                break
            # 1. 处理退出事件
            self._process_events()

            # 2. CARLA世界步进
            self.world.tick()

            # 3. 获取传感器数据
            try:
                image_rgb = self.rgb_queue.get(timeout=1.0)
                image_depth = self.depth_queue.get(timeout=1.0)
                inst_seg_image = self.inst_queue.get(timeout=1.0)
            except queue.Empty:
                logging.warning("传感器队列超时。")
                continue
                
            # 4. 图像预处理
            img_bgra_rgb = np.reshape(np.copy(image_rgb.raw_data), (self.height, self.width, 4))
            img_rgb = draw_carla_image(self.display, img_bgra_rgb)
            
            # 处理深度图
            depth_map = process_depth_image(image_depth)

            # 处理分割图
            inst_seg_bgra = np.reshape(np.copy(inst_seg_image.raw_data), (self.height, self.width, 4))
            semantic_labels, actor_ids_map = decode_instance_segmentation(inst_seg_bgra)
            
            # 5. 运行YOLO跟踪
            yolo_results = self.model.track(img_rgb, persist=True, verbose=False)
            
            if yolo_results:
                self.last_inference_time = yolo_results[0].speed.get('inference', 0.0)
            
            # 6. 获取真值投影矩阵
            world_2_camera = np.array(self.camera_rgb.get_transform().get_inverse_matrix())
            K = build_projection_matrix(self.width, self.height, self.camera_bp_rgb.get_attribute("fov").as_float())
            
            # 7. 融合、计算和绘制
            draw_yolo_and_truth(
                self.display, self.yolo_font, yolo_results, self.model,
                depth_map, actor_ids_map, self.world, self.ego_vehicle, 
                self.camera_bp_rgb, self.camera_rgb, K, world_2_camera,
                self.vehicle_tracker
            )

            # 8. 绘制模型帧率
            draw_fps_info(self.display, self.yolo_font, self.last_inference_time)
            
            # 9. 更新显示
            pygame.display.flip()
            self.clock.tick(TARGET_FPS)

    def run(self, duration=None):
        """
        按顺序执行所有操作的主方法
        
        参数:
            duration: 模拟持续时间（秒），如果为None则持续运行直到用户中断
        """
        try:
            self._init_pygame()
            self._load_yolo_model()
            self._setup_carla()
            self._spawn_actors()
            self.world.tick()
            self._game_loop(duration=duration)

        except KeyboardInterrupt:
            logging.info("用户中断...正在退出。")
        except Exception as e:
            logging.error(f"发生未捕获的错误: {e}", exc_info=True)
        finally:
            self.cleanup()

    def cleanup(self):
        """
        清理所有资源，确保正确关闭CARLA和Pygame
        """
        logging.info("正在清理资源...")

        original_settings = None
        if self.world:
            try:
                original_settings = self.world.get_settings()
            except Exception as e:
                logging.error(f"获取世界设置失败: {e}")

        try:
            # 恢复异步模式
            if self.world and original_settings:
                original_settings.synchronous_mode = False
                original_settings.fixed_delta_seconds = None
                self.world.apply_settings(original_settings)
                logging.info("CARLA 异步模式已恢复。")
            if self.traffic_manager:
                self.traffic_manager.set_synchronous_mode(False)
        except Exception as e:
            logging.warning(f"恢复异步模式失败: {e}")

        # 停止行人AI
        logging.info("正在停止行人AI...")
        for controller in self.walker_controllers:
            if controller and controller.is_alive:
                try:
                    controller.stop()
                except Exception as e:
                    logging.warning(f"停止行人控制器 {controller.id} 失败: {e}")

        # 销毁所有生成的actors
        logging.info(f"正在销毁 {len(self.actors_to_destroy)} 个 actors...")
        if self.client:
            # 先停止所有传感器
            for actor in self.actors_to_destroy:
                if actor and actor.is_alive and hasattr(actor, 'is_listening') and actor.is_listening:
                    actor.stop()
            
            # 批量销毁actors
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actors_to_destroy if x and x.is_alive])
        
        # 清空列表
        self.actors_to_destroy.clear()
        self.walker_controllers.clear()

        # 退出Pygame
        pygame.quit()
        logging.info('完成。')
