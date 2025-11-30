import carla
import queue
import pygame
import numpy as np
import logging
import random
from pygame.locals import K_ESCAPE

try:
    from ultralytics import YOLO
except ImportError:
    raise RuntimeError('cannot import ultralytics, please run "pip install ultralytics"')

# 导入自定义模块
import utils
import tracker
import visualization
import traffic_light  # 导入新模块

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
        
        # 传感器
        self.camera_rgb = None
        self.camera_depth = None
        self.inst_camera = None
        self.camera_bp_rgb = None
        
        # 队列
        self.rgb_queue = queue.Queue()
        self.depth_queue = queue.Queue()
        self.inst_queue = queue.Queue()
        
        # 实体管理
        self.traffic_manager = None
        self.actors_to_destroy = []
        self.walker_controllers = []
        
        # 模型与算法
        self.model = None
        self.last_inference_time = 0.0
        self.vehicle_tracker = tracker.VehicleTracker(dt=0.05)
        
        # 红绿灯检测方法
        self.tl_detection_method = args.tl_method if hasattr(args, 'tl_method') else 'deep_learning'
        # 确保方法有效
        if self.tl_detection_method not in ['deep_learning', 'hsv']:
            logging.warning(f"无效的红绿灯检测方法 '{self.tl_detection_method}'，使用默认值 'deep_learning'")
            self.tl_detection_method = 'deep_learning'
        
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
        pygame.display.set_caption("CARLA (YOLO: 车辆3D速度 + 红绿灯识别)")
        self.clock = pygame.time.Clock()
        self.yolo_font = pygame.font.SysFont("Arial", 16)

    def _load_yolo_model(self):
        """加载 YOLOv11 模型"""
        logging.info(f"正在从 {self.args.yolo_model} 加载 YOLO11 模型...")
        try:
            self.model = YOLO(self.args.yolo_model)
            logging.info(f"YOLO11 模型加载成功。")
            logging.info(f"目标车辆类别: {utils.YOLO_TARGET_CLASSES}")
            logging.info(f"目标交通设施: {traffic_light.YOLO_TARGET_CLASS}")
        except Exception as e:
            logging.error(f"加载 YOLO 模型失败: {e}")
            raise

    def _setup_carla(self):
        """连接 CARLA 并设置世界"""
        logging.info("正在连接 CARLA...")
        self.client = carla.Client(self.args.host, self.args.port)
        self.client.set_timeout(10.0)

        logging.info("正在加载地图 Town05...")
        self.world = self.client.load_world('Town05')

        trafficmanager_port = self.args.tm_port
        logging.info(f"正在连接到交通管理器 (端口: {trafficmanager_port})...")
        self.traffic_manager = self.client.get_trafficmanager(trafficmanager_port)
        self.traffic_manager.set_synchronous_mode(True)

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        logging.info("CARLA 同步模式已启用 (20 FPS)。")
        
        self.world.tick()

    def _spawn_actors(self):
        """生成主角、传感器和NPC"""
        logging.info("正在生成 Actors...")
        bp_lib = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()
        
        if not spawn_points:
            raise RuntimeError("没有可用的生成点")

        # 1. 生成主角
        vehicle_bp = bp_lib.find('vehicle.tesla.cybertruck')
        vehicle_bp.set_attribute('role_name', 'ego')
        transform = random.choice(spawn_points)
        spawn_points.remove(transform)
        
        self.ego_vehicle = self.world.try_spawn_actor(vehicle_bp, transform)
        if not self.ego_vehicle:
            raise RuntimeError("无法生成主角车辆")

        self.ego_vehicle.set_autopilot(True, self.traffic_manager.get_port())
        self.traffic_manager.vehicle_percentage_speed_difference(self.ego_vehicle, -30)
        self.actors_to_destroy.append(self.ego_vehicle)

        # 2. 生成传感器
        fov = 70.0
        camera_init_trans = carla.Transform(carla.Location(x=1.2, z=2.0))
        
        # RGB
        self.camera_bp_rgb = bp_lib.find('sensor.camera.rgb')
        self.camera_bp_rgb.set_attribute('image_size_x', str(self.width))
        self.camera_bp_rgb.set_attribute('image_size_y', str(self.height))
        self.camera_bp_rgb.set_attribute('fov', str(fov))
        self.camera_rgb = self.world.spawn_actor(self.camera_bp_rgb, camera_init_trans, attach_to=self.ego_vehicle)
        self.camera_rgb.listen(self.rgb_queue.put)
        self.actors_to_destroy.append(self.camera_rgb)

        # Depth
        camera_bp_depth = bp_lib.find('sensor.camera.depth')
        camera_bp_depth.set_attribute('image_size_x', str(self.width))
        camera_bp_depth.set_attribute('image_size_y', str(self.height))
        camera_bp_depth.set_attribute('fov', str(fov))
        self.camera_depth = self.world.spawn_actor(camera_bp_depth, camera_init_trans, attach_to=self.ego_vehicle)
        self.camera_depth.listen(self.depth_queue.put)
        self.actors_to_destroy.append(self.camera_depth)
        
        # Instance Segmentation
        inst_camera_bp = bp_lib.find('sensor.camera.instance_segmentation')
        inst_camera_bp.set_attribute('image_size_x', str(self.width))
        inst_camera_bp.set_attribute('image_size_y', str(self.height))
        inst_camera_bp.set_attribute('fov', str(fov))
        self.inst_camera = self.world.spawn_actor(inst_camera_bp, camera_init_trans, attach_to=self.ego_vehicle)
        self.inst_camera.listen(self.inst_queue.put)
        self.actors_to_destroy.append(self.inst_camera)
        
        # 3. 生成 NPC
        self._spawn_npcs(bp_lib, spawn_points)

    def _spawn_npcs(self, bp_lib, spawn_points):
        """生成 NPC 车辆和行人"""
        vehicle_blueprints = bp_lib.filter('vehicle.*')
        for _ in range(self.args.num_vehicles):
            if not spawn_points: break
            bp = random.choice(vehicle_blueprints)
            transform = random.choice(spawn_points)
            spawn_points.remove(transform)
            npc = self.world.try_spawn_actor(bp, transform)
            if npc:
                self.actors_to_destroy.append(npc)
                npc.set_autopilot(True, self.traffic_manager.get_port())

        walker_blueprints = bp_lib.filter('walker.pedestrian.*')
        controller_bp = bp_lib.find('controller.ai.walker')
        for _ in range(self.args.num_walkers):
            loc = self.world.get_random_location_from_navigation()
            if not loc: break
            walker = self.world.try_spawn_actor(random.choice(walker_blueprints), carla.Transform(loc))
            if walker:
                self.actors_to_destroy.append(walker)
                controller = self.world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
                if controller:
                    self.walker_controllers.append(controller)
                    self.actors_to_destroy.append(controller)
                    controller.start()
                    controller.go_to_location(self.world.get_random_location_from_navigation())
                    controller.set_max_speed(1 + random.random())

    def _process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.run_simulation = False
            elif event.type == pygame.KEYUP and event.key == K_ESCAPE:
                self.run_simulation = False

    def _game_loop(self):
        """核心游戏循环"""
        logging.info("开始 YOLOv11 多任务检测 (车辆 + 红绿灯)...")
        while self.run_simulation:
            self._process_events()
            self.world.tick()

            try:
                image_rgb = self.rgb_queue.get(timeout=1.0)
                image_depth = self.depth_queue.get(timeout=1.0)
                inst_seg_image = self.inst_queue.get(timeout=1.0)
            except queue.Empty:
                continue
                
            # --- 1. 图像预处理 ---
            img_bgra_rgb = np.reshape(np.copy(image_rgb.raw_data), (self.height, self.width, 4))
            img_rgb = utils.draw_carla_image(self.display, img_bgra_rgb)
            depth_map = utils.process_depth_image(image_depth)
            
            inst_seg_bgra = np.reshape(np.copy(inst_seg_image.raw_data), (self.height, self.width, 4))
            _, actor_ids_map = utils.decode_instance_segmentation(inst_seg_bgra)
            
            # --- 2. YOLO 统一检测 ---
            # 同时检测车辆和红绿灯
            yolo_results = self.model.track(img_rgb, persist=True, verbose=False)
            if yolo_results:
                self.last_inference_time = yolo_results[0].speed.get('inference', 0.0)
            
            # 投影矩阵 (用于 3D 框绘制)
            world_2_camera = np.array(self.camera_rgb.get_transform().get_inverse_matrix())
            K = utils.build_projection_matrix(self.width, self.height, self.camera_bp_rgb.get_attribute("fov").as_float())
            
            # --- 3. 任务分发与绘制 ---
            
            # 任务 A: 车辆检测与测速 (使用 visualization 中的新函数名)
            visualization.draw_vehicles_and_truth(
                self.display, self.yolo_font, yolo_results, self.model,
                depth_map, actor_ids_map, self.world, self.ego_vehicle, 
                self.camera_bp_rgb, self.camera_rgb, K, world_2_camera,
                self.vehicle_tracker
            )
            
            # 任务 B: 红绿灯检测与颜色识别 (新增)
            visualization.draw_traffic_lights(
                self.display, self.yolo_font, yolo_results, self.model, img_rgb, depth_map,
                detection_method=self.tl_detection_method
            )

            # --- 4. 绘制帧率 ---
            time_text = f"Inference: {self.last_inference_time:.1f} ms"
            self.display.blit(self.yolo_font.render(time_text, True, (255, 255, 0)), (10, 10))
            
            pygame.display.flip()
            self.clock.tick(20)

    def run(self):
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
            logging.error(f"发生错误: {e}", exc_info=True)
        finally:
            self.cleanup()

    def cleanup(self):
        logging.info("正在清理资源...")
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)

        for controller in self.walker_controllers:
            controller.stop()

        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actors_to_destroy if x and x.is_alive])
        pygame.quit()