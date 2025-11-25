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

# --- (新增) 图像处理库 ---
try:
    import cv2
except ImportError:
    raise RuntimeError('需要 OpenCV, 请运行 "pip install opencv-python"')

# --- YOLOv11 Import ---
try:
    from ultralytics import YOLO
except ImportError:
    raise RuntimeError('cannot import ultralytics, please run "pip install ultralytics"')

from pygame.locals import K_ESCAPE

# 我们的目标现在是红绿灯
YOLO_TARGET_CLASS = 'traffic light'

# --- (新增) 颜色检测的 HSV 阈值 ---
# 注意: HSV 在 OpenCV 中: H (0-179), S (0-255), V (0-255)

# 红色 (由于 H 环绕, 需要两个范围)
RED_LOWER_1 = np.array([0, 120, 120])
RED_UPPER_1 = np.array([10, 255, 255])
RED_LOWER_2 = np.array([170, 120, 120])
RED_UPPER_2 = np.array([179, 255, 255])

# 黄色
YELLOW_LOWER = np.array([20, 120, 120])
YELLOW_UPPER = np.array([35, 255, 255])

# 绿色
GREEN_LOWER = np.array([40, 120, 120])
GREEN_UPPER = np.array([90, 255, 255])

# 颜色检测结果的映射
CV_COLOR_MAP = {
    "RED": (255, 0, 0),
    "YELLOW": (255, 255, 0),
    "GREEN": (0, 255, 0),
    "UNKNOWN": (128, 128, 128)
}

# ==============================================================================
# -- (已删除) 3D->2D 投影辅助函数 (不再需要) -----------------------------
# ==============================================================================


# ==============================================================================
# -- (新增) 纯 CV 颜色检测算法 -----------------------------------------------
# ==============================================================================

def get_traffic_light_color(image_rgb, bbox):
    """
    分析 YOLO 边界框内的 RGB 图像，并返回检测到的颜色。
    使用 OpenCV (cv2) 和 HSV 颜色空间。
    """
    
    # 1. 从主图像中裁剪出红绿灯区域 (ROI)
    x1, y1, x2, y2 = bbox
    roi_rgb = image_rgb[y1:y2, x1:x2]

    # 2. 检查 ROI 是否有效
    if roi_rgb.shape[0] == 0 or roi_rgb.shape[1] == 0:
        return "UNKNOWN", CV_COLOR_MAP["UNKNOWN"]

    # 3. 将 ROI 转换为 HSV 颜色空间
    roi_hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)

    # 4. 创建每种颜色的掩码
    # (我们只关心高饱和度和高亮度的像素)
    mask_red1 = cv2.inRange(roi_hsv, RED_LOWER_1, RED_UPPER_1)
    mask_red2 = cv2.inRange(roi_hsv, RED_LOWER_2, RED_UPPER_2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    mask_yellow = cv2.inRange(roi_hsv, YELLOW_LOWER, YELLOW_UPPER)
    mask_green = cv2.inRange(roi_hsv, GREEN_LOWER, GREEN_UPPER)

    # 5. 计算每种颜色有多少亮起的像素
    red_pixels = np.count_nonzero(mask_red)
    yellow_pixels = np.count_nonzero(mask_yellow)
    green_pixels = np.count_nonzero(mask_green)

    # 6. 找出哪个颜色的像素最多
    # (设置一个最小阈值，比如 5 个像素，以避免噪点)
    pixel_counts = {"RED": red_pixels, "YELLOW": yellow_pixels, "GREEN": green_pixels}
    max_color = "UNKNOWN"
    max_pixels = 5 # 最小像素阈值
    
    for color, count in pixel_counts.items():
        if count > max_pixels:
            max_pixels = count
            max_color = color
            
    return max_color, CV_COLOR_MAP[max_color]

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

def draw_yolo_bboxes(surface, font, yolo_results, yolo_model, img_rgb):
    """
    (已更新) 在 Pygame surface 上绘制 YOLOv11 检测到的红绿灯
    并调用 CV 算法判断颜色
   
    """
    
    for r in yolo_results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            # 1. 确保 YOLO 检测到的是 'traffic light'
            if yolo_model.names.get(class_id) == YOLO_TARGET_CLASS:
                conf = float(box.conf[0])
                if conf < 0.5:
                    continue
                
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                
                # --- 2. (已修改) 调用 CV 算法判断颜色 ---
                # 我们不再需要 projected_lights
                # 我们传入 img_rgb 和方框坐标
                color_str, box_color = get_traffic_light_color(img_rgb, xyxy)
                
                # --- 3. 绘制 (使用 CV 算法返回的颜色和标签) ---
                x1, y1, x2, y2 = xyxy
                pygame.draw.rect(surface, box_color, pygame.Rect(x1, y1, x2 - x1, y2 - y1), 2)
                
                label = f"{YOLO_TARGET_CLASS} ({color_str}) {conf:.2f}"
                text_surface = font.render(label, True, (255, 255, 255), box_color)
                text_rect = text_surface.get_rect(bottomleft=(x1, y1 - 5))
                surface.blit(text_surface, text_rect)


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
        self.camera = None
        # self.camera_bp = None # <--- (已删除) 不再需要
        self.image_queue = queue.Queue()
        
        self.traffic_manager = None
        self.actors_to_destroy = []
        self.walker_controllers = []
        
        self.model = None
        
        self.last_inference_time = 0.0

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
        pygame.display.set_caption("CARLA YOLOv11 纯 CV 颜色检测 (20 FPS)")
        self.clock = pygame.time.Clock()
        self.yolo_font = pygame.font.SysFont("Arial", 16)

    def _load_yolo_model(self):
        """加载 YOLOv11 模型"""
        logging.info(f"正在从 {self.args.yolo_model} 加载 YOLO11 模型...")
        try:
            self.model = YOLO(self.args.yolo_model)
            logging.info(f"YOLO11 模型加载成功。")
            if YOLO_TARGET_CLASS not in self.model.names.values():
                logging.warning(f"警告: 你的 YOLO 模型不包含名为 '{YOLO_TARGET_CLASS}' 的类别。")
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
        settings.fixed_delta_seconds = 0.05  # 1.0 / 0.05 = 20 FPS
        self.world.apply_settings(settings)
        logging.info("CARLA 同步模式已启用 (20 FPS)。")
        
        self.world.tick()

    def _spawn_actors(self):
        """生成主角、NPC车辆和行人"""
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

        # --- 2. 生成 RGB 相机 (挂载到主角车) ---
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.width))
        camera_bp.set_attribute('image_size_y', str(self.height))
        camera_bp.set_attribute('fov', '70')
        
        camera_init_trans = carla.Transform(carla.Location(x=1.2, z=2))
        
        self.camera = self.world.spawn_actor(
            camera_bp, 
            camera_init_trans, 
            attach_to=self.ego_vehicle
        )
        self.actors_to_destroy.append(self.camera)
        self.camera.listen(self.image_queue.put)
        logging.info(f"已生成相机 (ID: {self.camera.id})")
        
        # --- (已删除) 不再需要实例分割摄像头 ---
        # inst_camera_bp = ...

        # --- 3. (新增) 生成 NPC 车辆 ---
        logging.info(f"正在生成 {self.args.num_vehicles} 辆 NPC 车辆...")
        vehicle_blueprints = bp_lib.filter('vehicle.*')
        
        for _ in range(self.args.num_vehicles):
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

        # --- 4. (新增) 生成 NPC 行人 ---
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
        logging.info("开始 YOLOv11 目标检测...")
        while self.run_simulation:
            # 1. 处理退出事件
            self._process_events()

            # 2. CARLA 世界步进
            self.world.tick()

            # 3. 获取传感器数据
            try:
                image = self.image_queue.get(timeout=1.0)
            except queue.Empty:
                logging.warning("传感器队列超时。")
                continue
                
            img_bgra = np.reshape(np.copy(image.raw_data), (self.height, self.width, 4))

            # 4. 绘制背景并获取 RGB 图像
            img_rgb = draw_carla_image(self.display, img_bgra)

            # --- 5. 运行 YOLOv11 目标检测 ---
            yolo_results = self.model(img_rgb, verbose=False)
            
            # --- 6. 提取检测用时 ---
            if yolo_results:
                self.last_inference_time = yolo_results[0].speed.get('inference', 0.0)
            
            # --- 7. (已删除) 获取红绿灯真实状态 (不再需要) ---
            # projected_lights = [] ...
            
            # 8. 绘制 YOLO 检测结果 (现在传入 img_rgb)
            draw_yolo_bboxes(self.display, self.yolo_font, yolo_results, self.model, img_rgb)

            # --- 9. 绘制模型帧率 ---
            time_text = f"YOLO Inference: {self.last_inference_time:.1f} ms"
            time_surface = self.yolo_font.render(time_text, True, (255, 255, 0)) # 黄色
            self.display.blit(time_surface, (10, 10))
            
            # 10. 更新显示
            pygame.display.flip()
            
            # 匹配 20 FPS
            self.clock.tick(20)

    def run(self):
        """按顺序执行所有操作"""
        try:
            self._init_pygame()
            self._load_yolo_model()
            self._setup_carla()
            self._spawn_actors()
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
        for actor in reversed(self.actors_to_destroy):
            if actor and actor.is_alive:
                try:
                    if hasattr(actor, 'is_listening') and actor.is_listening:
                        actor.stop()
                    actor.destroy()
                except Exception as e:
                    logging.warning(f"销毁 actor {actor.id} ({actor.type_id}) 失败: {e}")
        
        self.actors_to_destroy.clear()
        self.walker_controllers.clear()

        pygame.quit()
        logging.info('完成。')


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================

def main():
    # --- 参数解析 ---
    argparser = argparse.ArgumentParser(description='CARLA YOLOv11 纯 CV 颜色检测')
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='主机IP (default: 127.0.0.1)')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP端口 (default: 2000)')
    argparser.add_argument('--res', metavar='WIDTHxHEIGHT', default='1280x720', help='窗口分辨率 (default: 1280x720)')
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
    print('CARLA YOLOv11 纯 CV 颜色检测 (20 FPS) 说明:')
    print('ESC  : 退出')
    print('正在启动...')
    main()