import numpy as np
import matplotlib.pyplot as plt
from src.mcp_controller import Vehicle
from env import Env, draw_waypoints

import sys
import pathlib
import queue
import cv2
import os

# --- 添加路径以导入 Traffic_detection 模块 ---
traffic_detection_path = pathlib.Path(__file__).parent.parent / 'Traffic_detection'
sys.path.insert(0, str(traffic_detection_path))
sys.path.insert(0, str(pathlib.Path(__file__).with_name('src')))

# --- 尝试导入依赖 ---
try:
    from ultralytics import YOLO
    import traffic_light
    YOLO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: YOLO or traffic_light module not found. {e}")
    YOLO_AVAILABLE = False

try:
    import torch
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch/Torchvision not found. YOLOP will be disabled.")
    TORCH_AVAILABLE = False

from src.x_v2x_agent import Xagent
from src.global_route_planner import GlobalRoutePlanner

import time
import pygame
import carla

# --- 辅助函数：停止线检测 (基于 YOLOP 掩码) ---
# --- 修改处：增加 current_speed 参数 ---
def detect_stop_lines(lane_mask, current_speed=0.0):
    """
    检测路口停止线 (基于 YOLOP 掩码 + 速度动态阈值)
    Args:
        lane_mask: YOLOP 输出的车道线掩码
        current_speed: 当前车速 (km/h)
    """
    if lane_mask is None:
        return False

    height, width = lane_mask.shape[:2]
    edges = lane_mask

    mask = np.zeros_like(edges)
    # 梯形 ROI (关注车头前方)
    polygons = np.array([[
        (0, height), (width, height),
        (int(width * 0.7), int(height * 0.6)),
        (int(width * 0.3), int(height * 0.6))
    ]])
    cv2.fillPoly(mask, polygons, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 30, minLineLength=20, maxLineGap=20)
    stop_line_detected = False
    
    # --- 动态阈值计算 ---
    # 速度越快，阈值越小 (需要在停止线处于画面更上方/更远处时就反应)
    # 0 km/h -> 0.85 (近)
    # 50 km/h -> 0.50 (远)
    # 限制范围 [0.45, 0.85] 防止误判
    dynamic_threshold = 0.75 - (current_speed * 0.007)
    dynamic_threshold = np.clip(dynamic_threshold, 0.45, 0.75)
    # -------------------
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0: slope = 999.0
            else: slope = (y2 - y1) / (x2 - x1)
            
            # 停止线判定：
            # 1. 斜率水平
            # 2. Y坐标大于动态阈值
            if abs(slope) < 0.1 and min(y1, y2) > height * dynamic_threshold:
                stop_line_detected = True
                # 可选：打印调试信息看看阈值变化
                # print(f"Speed: {current_speed:.1f}, Thresh: {dynamic_threshold:.2f}, LineY: {min(y1, y2)/height:.2f}")
                break
    return stop_line_detected

# --- YOLOP 检测类 ---
class YOLOPDetector:
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = None
        
        if not TORCH_AVAILABLE:
            return

        # 1. 定位YOLOP
        yolop_local_path = pathlib.Path('YOLOP')
        
        try:
            print("Loading YOLOP model...")
            if yolop_local_path.exists():
                print(f"Found local YOLOP at {yolop_local_path}")
                self.model = torch.hub.load(str(yolop_local_path), 'yolop', source='local', pretrained=True)
                print(f"YOLOP model loaded from local repository.")
            else:
                print("Local YOLOP not found, trying to load from torch hub (internet required)...")
                self.model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
            
            self.model.to(self.device)
            self.model.eval()
            print("YOLOP loaded successfully.")
            
            # YOLOP 标准预处理
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
        except Exception as e:
            print(f"Failed to load YOLOP: {e}")
            self.model = None

    def infer(self, img_rgb):
        """
        执行 YOLOP 推理
        Returns:
            img_vis: 叠加了车道线和可行驶区域的图像
            ll_seg_mask: 车道线二值掩码 (uint8, 0或255)
            da_seg_mask: 可行驶区域二值掩码 (uint8, 0或255)
        """
        if self.model is None:
            return img_rgb, None, None

        img_h, img_w = img_rgb.shape[:2]
        
        # 1. 预处理
        img_resized = cv2.resize(img_rgb, (640, 640))
        input_tensor = self.transform(img_resized).unsqueeze(0).to(self.device)
        
        # 2. 推理
        with torch.no_grad():
            det_out, da_seg_out, ll_seg_out = self.model(input_tensor)
            
        # 3. 后处理 - 分割掩码
        # Drivable Area
        da_seg_mask = torch.nn.functional.interpolate(da_seg_out, size=(img_h, img_w), mode='bilinear', align_corners=True)
        da_seg_mask = torch.argmax(da_seg_mask, dim=1).squeeze().cpu().numpy() # 0 or 1
        da_seg_mask = (da_seg_mask * 255).astype(np.uint8)
        
        # Lane Line
        ll_seg_mask = torch.nn.functional.interpolate(ll_seg_out, size=(img_h, img_w), mode='bilinear', align_corners=True)
        ll_seg_mask = torch.argmax(ll_seg_mask, dim=1).squeeze().cpu().numpy() # 0 or 1
        ll_seg_mask = (ll_seg_mask * 255).astype(np.uint8)
        
        # 4. 可视化叠加
        img_vis = img_rgb.copy()
        
        # 绘制可行驶区域 (蓝色，半透明)
        color_area = np.zeros_like(img_vis)
        color_area[da_seg_mask > 0] = [0, 0, 255] # RGB: Blue
        
        mask_bool = (da_seg_mask > 0)
        if mask_bool.any():
            img_vis[mask_bool] = cv2.addWeighted(img_vis[mask_bool], 0.7, color_area[mask_bool], 0.3, 0).squeeze()
        
        # 绘制车道线 (绿色)
        # 这里的 ll_seg_mask 已经是二值化的边缘了，可以直接用来显示
        img_vis[ll_seg_mask > 100] = [0, 255, 0] # RGB: Green
        
        return img_vis, ll_seg_mask, da_seg_mask


class MPCCarSimulation:
    """MPC车辆控制仿真主类 (集成 YOLOP 车道/路面检测 + YOLOv11 红绿灯检测)"""
    
    def __init__(self):
        """初始化仿真参数和环境"""
        # ... (前面的参数保持不变) ...
        self.simulation_params = {
            'time_step': 0.05,
            'target_speed': 45,
            'sample_resolution': 2.0,
            'display_mode': "pygame",
            'max_simulation_steps': 5000, 
            'destination_threshold': 1.0, 
            'map_name': "Town05"       
        }
        
        # 初始化环境
        self.env = self._initialize_environment()
        self.spawn_points = self.env.map.get_spawn_points()
        
        # 仿真数据记录
        self.simulation_data = {
            'trajectory': [], 'velocities': [], 'accelerations': [], 'steerings': [], 'times': []
        }
        
        # 车辆和控制器
        self.vehicle = None
        self.agent = None
        self.route = None
        
        # AI 模型相关
        self.yolo_model = None      # YOLOv11 (红绿灯 - 长焦)
        self.yolop_detector = None  # YOLOP (车道/路面 - 广角)
        
        # --- 修改处：定义双目摄像头的队列和传感器变量 ---
        self.wide_queue = queue.Queue() # 广角队列
        self.tele_queue = queue.Queue() # 长焦队列
        self.wide_sensor = None
        self.tele_sensor = None
        # -------------------------------------------
        
        self._init_models()

    def _init_models(self):
        """初始化所有 AI 模型"""
        # 1. Init YOLOv11 (Traffic Lights)
        if YOLO_AVAILABLE:
            model_path = pathlib.Path("YoloModel/yolo11x.pt")
            try:
                print(f"Loading YOLOv11 model from {model_path}...")
                self.yolo_model = YOLO(str(model_path)) if model_path.exists() else YOLO("yolo11s.pt")
                print("YOLOv11 model loaded.")
            except Exception as e:
                print(f"Failed to load YOLOv11: {e}")
        
        # 2. Init YOLOP (Lane & Drivable Area)
        if TORCH_AVAILABLE:
            self.yolop_detector = YOLOPDetector()

    def _initialize_environment(self):
        """初始化仿真环境"""
        host = 'localhost'
        port = 2000
        target_map = self.simulation_params['map_name']
        
        try:
            client = carla.Client(host, port)
            client.set_timeout(10.0)
            world = client.get_world()
            current_map_name = world.get_map().name
            
            if target_map not in current_map_name:
                print(f"Loading {target_map}...")
                client.load_world(target_map)
            else:
                print(f"{target_map} is already loaded.")
                
        except Exception as e:
            print(f"Error checking/loading map: {e}")

        env = Env(
            host=host,
            port=port,
            display_method=self.simulation_params['display_mode'],
            dt=self.simulation_params['time_step']
        )
        env.clean()
        return env
    
    def _setup_sensors(self):
        """设置双目摄像头系统 (广角 + 长焦)"""
        if self.vehicle is None: return

        bp_library = self.env.world.get_blueprint_library()
        
        # --- 1. 广角摄像头 (Wide) ---
        bp_wide = bp_library.find('sensor.camera.rgb')
        bp_wide.set_attribute('image_size_x', '1280')
        bp_wide.set_attribute('image_size_y', '720')
        bp_wide.set_attribute('fov', '90')
        # 关键：设置为 0.0 确保每帧都生成数据
        bp_wide.set_attribute('sensor_tick', '0.0') 
        
        spawn_point_wide = carla.Transform(carla.Location(x=1.0, z=2.0))
        
        # --- 修改处：添加 attachment_type=carla.AttachmentType.Rigid ---
        self.wide_sensor = self.env.world.spawn_actor(
            bp_wide, spawn_point_wide, attach_to=self.env.ego_vehicle,
            attachment_type=carla.AttachmentType.Rigid
        )
        self.wide_sensor.listen(self.wide_queue.put)
        self.env.actor_list.append(self.wide_sensor)

        # --- 2. 长焦摄像头 (Tele) ---
        bp_tele = bp_library.find('sensor.camera.rgb')
        bp_tele.set_attribute('image_size_x', '1280')
        bp_tele.set_attribute('image_size_y', '720')
        bp_tele.set_attribute('fov', '35')
        bp_tele.set_attribute('sensor_tick', '0.0')
        
        spawn_point_tele = carla.Transform(carla.Location(x=1.0, z=2.0))
        
        # --- 修改处：添加 attachment_type=carla.AttachmentType.Rigid ---
        self.tele_sensor = self.env.world.spawn_actor(
            bp_tele, spawn_point_tele, attach_to=self.env.ego_vehicle,
            attachment_type=carla.AttachmentType.Rigid
        )
        self.tele_sensor.listen(self.tele_queue.put)
        self.env.actor_list.append(self.tele_sensor)
        
        print("Dual Camera System initialized (Rigid Attachment).")

    def _setup_route(self, start_idx, end_idx):
        """规划全局路径"""
        route_planner = GlobalRoutePlanner(self.env.map, self.simulation_params['sample_resolution'])
        self.route = route_planner.trace_route(
            self.spawn_points[start_idx].location, 
            self.spawn_points[end_idx].location
        )
        draw_waypoints(self.env.world, [wp for wp, _ in self.route], z=0.5, color=(0, 255, 0))
        self.env.reset(spawn_point=self.spawn_points[start_idx])
        
    def _initialize_vehicle_and_agent(self, start_idx, end_idx):
        """初始化车辆和 MPC 智能体"""
        self.vehicle = Vehicle(
            actor=self.env.ego_vehicle, horizon=10, 
            target_v=self.simulation_params['target_speed'],
            delta_t=self.simulation_params['time_step'], max_iter=30
        )
        
        self.agent = Xagent(self.env, self.vehicle, dt=self.simulation_params['time_step'])
        self.agent.set_start_end_transforms(start_idx, end_idx)
        self.agent.plan_route(self.agent._start_transform, self.agent._end_transform)
    
    def _update_pygame_display(self, step, is_red=False, is_stop=False, brake_info="", yolo_data=None):
        """
        更新显示
        Args:
            step: 当前仿真步数
            is_red: 是否检测到红灯 (bool)
            is_stop: 是否检测到停止线 (bool)
            brake_info: 刹车状态附加信息 (str)
            yolo_data: (img_wide, img_tele) 元组
        """
        self.env.hud.tick(self.env, self.env.clock)
        if step == 0: self.env.display.fill((0, 0, 0))
        self.env.hud.render(self.env.display)
        
        # --- 画中画显示 (右侧) ---
        if yolo_data:
            img_wide, img_tele = yolo_data
            
            # 1. 广角视图 (Top)
            if img_wide is not None:
                display_wide = cv2.resize(img_wide, (640, 360))
                cv2.putText(display_wide, "Wide Angle (Lane)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                surf_wide = pygame.surfarray.make_surface(display_wide.swapaxes(0, 1))
                self.env.display.blit(surf_wide, (640, 0))
                pygame.draw.rect(self.env.display, (255, 255, 255), (640, 0, 640, 360), 2)

            # 2. 长焦视图 (Bottom)
            if img_tele is not None:
                display_tele = cv2.resize(img_tele, (640, 360))
                cv2.putText(display_tele, "Telephoto (Traffic Light)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                surf_tele = pygame.surfarray.make_surface(display_tele.swapaxes(0, 1))
                self.env.display.blit(surf_tele, (640, 360))
                pygame.draw.rect(self.env.display, (255, 255, 255), (640, 360, 640, 360), 2)
        
        # --- 修改处：左下角状态提示信息 ---
        font = pygame.font.SysFont("Arial", 25, bold=True)
        base_y = self.env.display.get_height() - 100 # 起始 Y 坐标 (左下角)
        
        # 1. 红灯提示 (红色)
        if is_red:
            text_red = font.render("WARNING: RED LIGHT DETECTED", True, (255, 0, 0))
            pygame.draw.rect(self.env.display, (0,0,0), (20, base_y, text_red.get_width()+10, 30)) # 黑色背景衬托
            self.env.display.blit(text_red, (25, base_y))
            base_y += 35 # 下一行位置
            
        # 2. 停止线提示 (青色)
        if is_stop:
            text_line = font.render("INFO: STOP LINE DETECTED", True, (0, 255, 255))
            pygame.draw.rect(self.env.display, (0,0,0), (20, base_y, text_line.get_width()+10, 30))
            self.env.display.blit(text_line, (25, base_y))
            base_y += 35

        # 3. 刹车提示 (黄色)
        if brake_info:
            text_brake = font.render(f"ACTION: {brake_info}", True, (255, 255, 0))
            pygame.draw.rect(self.env.display, (0,0,0), (20, base_y, text_brake.get_width()+10, 30))
            self.env.display.blit(text_brake, (25, base_y))
        # --------------------------------

        pygame.display.flip()
        self.env.check_quit()

    def _process_yolo_detection(self):
        """
        双流感知处理：加入车速读取，实现动态停止线检测
        """
        is_red_light = False
        info_text = ""
        img_wide_vis = None
        img_tele_vis = None
        tl_results = {}
        stop_line_detected = False
        
        def get_latest_frame(q, timeout=2.0):
            try:
                data = q.get(timeout=timeout)
                while not q.empty():
                    try:
                        data = q.get_nowait()
                    except queue.Empty:
                        break
                return data
            except queue.Empty:
                return None
        
        data_wide = get_latest_frame(self.wide_queue)
        data_tele = get_latest_frame(self.tele_queue)
        
        if data_wide is None or data_tele is None:
            return is_red_light, info_text, None, None, {}, False

        try:
            # --- 1. 获取当前车速 (km/h) ---
            # 如果车辆还未生成，速度为0
            current_speed = 0.0
            if self.env.ego_vehicle:
                v = self.env.ego_vehicle.get_velocity()
                current_speed = 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2)
            # ---------------------------

            # --- 处理广角图像 (Wide) ---
            array_wide = np.frombuffer(data_wide.raw_data, dtype=np.dtype("uint8"))
            array_wide = np.reshape(array_wide, (data_wide.height, data_wide.width, 4))
            raw_wide = array_wide[:, :, :3][:, :, ::-1].copy()
            img_wide_vis = raw_wide.copy()
            
            ll_mask = None
            if self.yolop_detector:
                img_wide_vis, ll_mask, _ = self.yolop_detector.infer(raw_wide)
            
            # --- 修改处：传入车速进行停止线检测 ---
            if ll_mask is not None:
                stop_line_detected = detect_stop_lines(ll_mask, current_speed)
            # -----------------------------------

            # --- 处理长焦图像 (Tele) ---
            array_tele = np.frombuffer(data_tele.raw_data, dtype=np.dtype("uint8"))
            array_tele = np.reshape(array_tele, (data_tele.height, data_tele.width, 4))
            raw_tele = array_tele[:, :, :3][:, :, ::-1].copy()
            img_tele_vis = raw_tele.copy()

            # ROI 裁剪 (Top 30%, Center 30%)
            h, w = raw_tele.shape[:2]
            roi_h_end = int(h * 0.3)
            roi_w_start = int(w * 0.35)
            roi_w_end = int(w * 0.65)
            
            tele_roi = raw_tele[0:roi_h_end, roi_w_start:roi_w_end]

            if self.yolo_model and tele_roi.size > 0:
                results = self.yolo_model(tele_roi, verbose=False)
                tl_results = traffic_light.detect_traffic_lights(
                    tele_roi, results, self.yolo_model.names, method='hsv'
                )
                
                for bbox, info in tl_results.items():
                    color = info['color']
                    x1_roi, y1_roi, x2_roi, y2_roi = bbox
                    
                    x1 = x1_roi + roi_w_start
                    x2 = x2_roi + roi_w_start
                    y1 = y1_roi
                    y2 = y2_roi
                    
                    if color == 'RED':
                        if (y2 - y1) > 10: 
                            is_red_light = True
                            info_text = "RED LIGHT!"
                    
                    c_rgb = (0, 0, 255) if color=='RED' else ((0, 255, 0) if color=='GREEN' else (255, 255, 0))
                    cv2.rectangle(img_tele_vis, (x1, y1), (x2, y2), c_rgb, 2)
                    cv2.putText(img_tele_vis, color, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c_rgb, 1)

            cv2.rectangle(img_tele_vis, (roi_w_start, 0), (roi_w_end, roi_h_end), (200, 200, 200), 1)

        except Exception as e:
            print(f"Detection error: {e}")
            import traceback
            traceback.print_exc()
            
        return is_red_light, info_text, img_wide_vis, img_tele_vis, tl_results, stop_line_detected

    def _is_vehicle_in_junction(self):
        """检查车辆是否在路口"""
        if self.env.ego_vehicle and self.env.map:
            loc = self.env.ego_vehicle.get_location()
            wp = self.env.map.get_waypoint(loc, project_to_road=True)
            return wp.is_junction
        return False
    
    def _record_simulation_data(self, step, next_state, acceleration, steering):
        """记录仿真数据"""
        x, y, _, vx, _, _ = next_state[0]
        current_time = step * self.simulation_params['time_step']
        
        self.simulation_data['trajectory'].append([x, y])
        self.simulation_data['velocities'].append(vx)
        self.simulation_data['accelerations'].append(acceleration)
        self.simulation_data['steerings'].append(steering)
        self.simulation_data['times'].append(current_time)
    
    def _check_destination_reached(self, next_state):
        """检查是否到达目的地"""
        x, y = next_state[0][0], next_state[0][1]
        end_x, end_y = self.agent._end_transform.location.x, self.agent._end_transform.location.y
        
        distance = np.linalg.norm([x - end_x, y - end_y])
        return distance < self.simulation_params['destination_threshold']
    
    def _visualize_results(self):
        """可视化仿真结果"""
        if not self.simulation_data['times']:
            return

        trajectory = np.array(self.simulation_data['trajectory'])
        velocities = np.array(self.simulation_data['velocities'])
        accelerations = np.array(self.simulation_data['accelerations'])
        steerings = np.array(self.simulation_data['steerings'])
        times = np.array(self.simulation_data['times'])
        
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        
        axs[0, 0].plot(trajectory[:, 0], trajectory[:, 1], 
                      label="Vehicle Path", color='darkorange', linewidth=2)
        axs[0, 0].scatter(self.agent._start_transform.location.x, 
                         self.agent._start_transform.location.y, 
                         color='green', label="Start", zorder=5)
        axs[0, 0].scatter(self.agent._end_transform.location.x, 
                         self.agent._end_transform.location.y, 
                         color='red', label="End", zorder=5)
        
        if self.route:
            route_points = np.array([[wp.transform.location.x, wp.transform.location.y] 
                                    for wp, _ in self.route])
            axs[0, 0].plot(route_points[:, 0], route_points[:, 1], 
                        '--', color='blue', label="Planned Route", alpha=0.6)
        
        axs[0, 0].set_title("Vehicle Path and Planned Route", fontsize=14)
        axs[0, 0].set_xlabel("X Position", fontsize=12)
        axs[0, 0].set_ylabel("Y Position", fontsize=12)
        axs[0, 0].legend(loc='upper left', fontsize=10)
        axs[0, 0].grid(True)
        
        axs[0, 1].plot(times, velocities, 
                      label="Velocity (m/s)", color='royalblue', linewidth=2)
        axs[0, 1].set_title("Velocity over Time", fontsize=14)
        axs[0, 1].set_ylabel("Velocity (m/s)", fontsize=12)
        axs[0, 1].legend(loc='upper right', fontsize=10)
        axs[0, 1].grid(True)
        
        axs[1, 0].plot(times, accelerations, 
                      label="Acceleration (m/s²)", color='orange', linewidth=2)
        axs[1, 0].set_title("Acceleration over Time", fontsize=14)
        axs[1, 0].set_xlabel("Time (s)", fontsize=12)
        axs[1, 0].set_ylabel("Acceleration (m/s²)", fontsize=12)
        axs[1, 0].legend(loc='upper right', fontsize=10)
        axs[1, 0].grid(True)
        
        axs[1, 1].plot(times, steerings, 
                      label="Steering Angle (rad)", color='green', linewidth=2)
        axs[1, 1].set_title("Steering Angle over Time", fontsize=14)
        axs[1, 1].set_xlabel("Time (s)", fontsize=12)
        axs[1, 1].set_ylabel("Steering Angle (rad)", fontsize=12)
        axs[1, 1].legend(loc='upper right', fontsize=10)
        axs[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def run_simulation(self, start_idx=80, end_idx=20):
        try:
            self._setup_route(start_idx, end_idx)
            self._initialize_vehicle_and_agent(start_idx, end_idx)
            if self.env.display_method == "pygame": self.env.init_display()
            self._setup_sensors()
            
            # 获取初始的步长参数 (通常是 1.5)
            # 注意：需确保 x_v2x_agent.py 中 self.dist_step 是可访问的
            cruise_speed = self.simulation_params['target_speed']
            original_dist_step = 1.5 # 或者 self.agent.dist_step

            for step in range(self.simulation_params['max_simulation_steps']):
                is_red, _, img_w, img_t, tl_res, is_stop = self._process_yolo_detection()
                
                # 先运行一步获取状态，或者放在后面也可以，关键是参数设置要在 run_step 之前
                
                brake_msg = ""
                # --- 刹车逻辑 ---
                if is_red and is_stop and not self._is_vehicle_in_junction():
                    # 1. 设置目标速度为 0 (现在 bounds 已经放开，这会生效了)
                    self.agent._model.set_target_velocity(0.0)
                    
                    # 2. 【关键】锁定参考轨迹，不让它随车身惯性前移
                    # 这样 MPC 才会为了消除位置误差而倾向于停在当前参考点
                    self.agent.dist_step = 0.0 
                    
                    brake_msg = "BRAKING (MPC Target=0)"
                else:
                    # 恢复正常行驶
                    self.agent._model.set_target_velocity(cruise_speed)
                    self.agent.dist_step = original_dist_step
                # ----------------
                
                acc, steer, next_state = self.agent.run_step()
                
                self._record_simulation_data(step, next_state, acc, steer)
                self.env.step([acc, steer])
                
                if self.env.display_method == "pygame":
                    yolo_data = (img_w, img_t) if img_w is not None else None
                    self._update_pygame_display(
                        step, 
                        is_red=is_red, 
                        is_stop=is_stop, 
                        brake_info=brake_msg,
                        yolo_data=yolo_data
                    )
                
                if self._check_destination_reached(next_state):
                    print("Destination reached!"); break
                    
        except KeyboardInterrupt:
            print("User interrupted.")
        except Exception as e:
            print(f"Simulation error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._visualize_results()

def main():
    sim = MPCCarSimulation()
    sim.run_simulation()

if __name__ == "__main__":
    main()