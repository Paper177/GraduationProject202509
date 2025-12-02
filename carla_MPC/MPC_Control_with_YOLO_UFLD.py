import numpy as np
import matplotlib.pyplot as plt
from src.mcp_controller import Vehicle
from env import Env, draw_waypoints

import sys
import pathlib
import queue
import cv2
import os

# --- 路径配置 ---
# 1. 添加 Traffic_detection 模块路径
traffic_detection_path = 'Traffic_detection'
sys.path.insert(0, str(traffic_detection_path))
sys.path.insert(0, str(pathlib.Path(__file__).with_name('src')))

# 2. 添加 Ultra-Fast-Lane-Detection 项目路径
ufld_project_path = 'carla_MPC/Ultra-Fast-Lane-Detection'
sys.path.insert(0, str(ufld_project_path))

# --- 导入依赖 ---
# 尝试导入 YOLO 和 traffic_light
try:
    from ultralytics import YOLO
    import traffic_light
    YOLO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: YOLO or traffic_light module not found. Traffic light detection disabled. {e}")
    YOLO_AVAILABLE = False

# 尝试导入 PyTorch 和 UFLD 相关依赖
try:
    import torch
    import torchvision.transforms as transforms
    from scipy import special
    # 尝试从 UFLD 项目中导入模型定义
    # 注意: 需要确保 'model.model' 在 sys.path 中可达
    from model.model import parsingNet
    UFLD_AVAILABLE = True
    print("Ultra-Fast-Lane-Detection dependencies loaded.")
except ImportError as e:
    print(f"Warning: Ultra-Fast-Lane-Detection not found or dependencies missing. Lane detection will be limited. {e}")
    UFLD_AVAILABLE = False

from src.x_v2x_agent import Xagent
from src.global_route_planner import GlobalRoutePlanner

import time
import pygame
import carla

# --- UFLD 车道线检测类 ---
class UFLDLaneDetector:
    def __init__(self, model_path, use_cuda=True):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        
        if not UFLD_AVAILABLE:
            self.model = None
            return

        # TuSimple 配置参数 (对应 tusimple_18.pth)
        self.cls_num_per_lane = 56
        self.griding_num = 100
        self.backbone = '18'
        self.row_anchor = [64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164, 168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216, 220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268, 272, 276, 280, 284]
        
        # 初始化模型
        try:
            self.model = parsingNet(pretrained=False, backbone=self.backbone, cls_dim=(self.griding_num + 1, self.cls_num_per_lane, 4), use_aux=False).to(self.device)
            
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict['model'])
                self.model.eval()
                print(f"UFLD model loaded from {model_path}")
            else:
                print(f"Error: UFLD weights not found at {model_path}")
                self.model = None
        except Exception as e:
            print(f"Failed to initialize UFLD model: {e}")
            self.model = None

        # 图像预处理
        self.img_transforms = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        self.img_w, self.img_h = 1280, 720 # 假设 CARLA 输出分辨率

    def detect_lanes(self, img_rgb):
        """
        执行车道线检测
        Args:
            img_rgb: 原始 RGB 图像 (numpy array)
        Returns:
            img_vis: 绘制了车道线的图像
        """
        if self.model is None:
            return img_rgb

        # 保存原始尺寸
        orig_h, orig_w = img_rgb.shape[:2]
        
        # 预处理
        # PIL Image 需要 RGB
        from PIL import Image
        img_pil = Image.fromarray(img_rgb)
        input_tensor = self.img_transforms(img_pil).unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            output = self.model(input_tensor)
            # output shape: (1, 101, 56, 4) if processed correctly, usually need parsing
            # UFLD output is classification result for each row anchor
            
            # 解析输出
            # parsing: (B, griding_num+1, cls_num_per_lane, lanes)
            # 这里的 output 取决于 model forward 的返回值，通常是 x
            
            # 根据 UFLD demo 逻辑处理:
            processed_output = output[0] if isinstance(output, tuple) else output
            
            col_sample = np.linspace(0, 800 - 1, self.griding_num)
            col_sample_w = col_sample[1] - col_sample[0]

            cls_out_ext = processed_output.data.cpu().numpy() # (1, 101, 56, 4)
            
            out_j = cls_out_ext[0] # (101, 56, 4)
            out_j = out_j[:, ::-1, :] # 翻转行顺序? 根据 UFLD 源码逻辑
            
            prob = special.softmax(out_j[:-1, :, :], axis=0)
            idx = np.arange(self.griding_num) + 1
            idx = idx.reshape(-1, 1, 1)
            loc = np.sum(prob * idx, axis=0)
            out_j = np.argmax(out_j, axis=0)
            
            loc[out_j == self.griding_num] = 0
            out_j = loc

            # 可视化
            img_vis = img_rgb.copy()
            
            # 遍历 4 条车道线
            for i in range(out_j.shape[1]):
                if np.sum(out_j[:, i] != 0) > 2: # 如果该车道线有效点大于2个
                    points = []
                    for k in range(out_j.shape[0]):
                        if out_j[k, i] > 0:
                            # 映射回原图坐标
                            # UFLD internal size: 800x288
                            # Row anchors are relative to 288 height
                            
                            # 计算 x 坐标: (loc * col_sample_w * orig_w) / 800 - 1
                            original_x = int(out_j[k, i] * col_sample_w * orig_w / 800) - 1
                            
                            # 计算 y 坐标: row_anchor 是相对于 288 的 y 坐标
                            # row_anchor list is [64, ... 284]
                            original_y = int(self.row_anchor[self.cls_num_per_lane - 1 - k] * orig_h / 288) - 1
                            
                            points.append((original_x, original_y))
                    
                    # 绘制连线
                    for j in range(len(points) - 1):
                        cv2.line(img_vis, points[j], points[j+1], (0, 255, 0), 3) # 绿色车道线

        return img_vis

# --- 简单的停止线检测函数 (保留作为辅助) ---
def detect_stop_lines(img_rgb):
    """检测路口停止线 (保留原来的简单CV逻辑用于停车辅助)"""
    height, width = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    mask = np.zeros_like(edges)
    polygons = np.array([[
        (0, height), (width, height),
        (int(width * 0.7), int(height * 0.6)),
        (int(width * 0.3), int(height * 0.6))
    ]])
    cv2.fillPoly(mask, polygons, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 30, minLineLength=30, maxLineGap=20)
    stop_line_detected = False
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0: slope = 999.0
            else: slope = (y2 - y1) / (x2 - x1)
            
            # 停止线判定：水平且靠下
            if abs(slope) < 0.1 and min(y1, y2) > height * 0.65:
                stop_line_detected = True
                break
                
    return stop_line_detected


class MPCCarSimulation:
    """MPC车辆控制仿真主类 (集成YOLO红绿灯检测与UFLD车道线检测)"""
    
    def __init__(self):
        """初始化仿真参数和环境"""
        # 仿真参数配置
        self.simulation_params = {
            'time_step': 0.05,         # 仿真时间步长（秒）
            'target_speed': 30,        # 目标速度（km/h）
            'sample_resolution': 2.0,  # 路径规划采样分辨率
            'display_mode': "pygame",  # 显示模式："spec" 或 "pygame"
            'max_simulation_steps': 5000, # 最大仿真步数
            'destination_threshold': 1.0, # 到达目的地的距离阈值（米）
            'map_name': "Town03"       # 地图名称配置
        }
        
        # 初始化环境
        self.env = self._initialize_environment()
        self.spawn_points = self.env.map.get_spawn_points()
        
        # 仿真数据记录
        self.simulation_data = {
            'trajectory': [],
            'velocities': [],
            'accelerations': [],
            'steerings': [],
            'times': []
        }
        
        # 车辆和控制器
        self.vehicle = None
        self.agent = None
        self.route = None
        
        # YOLO & UFLD 相关
        self.yolo_model = None
        self.lane_detector = None
        self.rgb_queue = queue.Queue()
        self.yolo_sensor = None
        self._init_models()

    def _init_models(self):
        """初始化 YOLO 和 UFLD 模型"""
        # 1. Init YOLO
        if YOLO_AVAILABLE:
            model_path = pathlib.Path("YoloModel/yolo11x.pt")
            try:
                print(f"Loading YOLO model from {model_path}...")
                self.yolo_model = YOLO(str(model_path)) if model_path.exists() else YOLO("yolo11s.pt")
                print("YOLO model loaded successfully.")
            except Exception as e:
                print(f"Failed to load YOLO model: {e}")
                self.yolo_model = None
        
        # 2. Init UFLD
        if UFLD_AVAILABLE:
            # 假设权重在 Traffic_detection 目录下
            ufld_weights = "carla_MPC/Ultra-Fast-Lane-Detection/configs/tusimple_18.pth"
            print(f"Initializing UFLD Lane Detector with weights: {ufld_weights}")
            self.lane_detector = UFLDLaneDetector(str(ufld_weights))

    def _initialize_environment(self):
        """初始化仿真环境，并根据配置加载地图"""
        host = 'localhost'
        port = 2000
        target_map = self.simulation_params['map_name']
        
        try:
            client = carla.Client(host, port)
            client.set_timeout(10.0)
            world = client.get_world()
            current_map_name = world.get_map().name
            
            if target_map not in current_map_name:
                print(f"Current map is {current_map_name}. Loading {target_map}...")
                client.load_world(target_map)
                print(f"{target_map} loaded successfully.")
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
    
    def _setup_yolo_sensor(self):
        """设置用于检测的 RGB 摄像头"""
        if self.vehicle is None:
            return

        bp_library = self.env.world.get_blueprint_library()
        camera_bp = bp_library.find('sensor.camera.rgb')
        # 设置分辨率 (640x480 适合实时处理, 也可以更高但会影响 FPS)
        # 注意: UFLD 会 resize 到 800x288, YOLO 也会 resize
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '45')
        
        # 安装在车顶位置
        spawn_point = carla.Transform(carla.Location(x=1.0, z=2.0))
        
        self.yolo_sensor = self.env.world.spawn_actor(
            camera_bp, 
            spawn_point, 
            attach_to=self.env.ego_vehicle
        )
        self.yolo_sensor.listen(self.rgb_queue.put)
        self.env.actor_list.append(self.yolo_sensor)
        print("Detection sensor initialized.")

    def _setup_route(self, start_idx, end_idx):
        """设置起点和终点，规划全局路径"""
        route_planner = GlobalRoutePlanner(
            self.env.map, 
            self.simulation_params['sample_resolution']
        )
        
        self.route = route_planner.trace_route(
            self.spawn_points[start_idx].location, 
            self.spawn_points[end_idx].location
        )
        
        draw_waypoints(
            self.env.world, 
            [wp for wp, _ in self.route], 
            z=0.5, 
            color=(0, 255, 0)
        )
        
        self.env.reset(spawn_point=self.spawn_points[start_idx])
        
    def _initialize_vehicle_and_agent(self, start_idx, end_idx):
        """初始化车辆动力学模型和智能体"""
        self.vehicle = Vehicle(
            actor=self.env.ego_vehicle,
            horizon=10,
            target_v=self.simulation_params['target_speed'],
            delta_t=self.simulation_params['time_step'],
            max_iter=30
        )
        
        self.agent = Xagent(
            self.env, 
            self.vehicle, 
            dt=self.simulation_params['time_step']
        )
        self.agent.set_start_end_transforms(start_idx, end_idx)
        self.agent.plan_route(self.agent._start_transform, self.agent._end_transform)
    
    def _update_pygame_display(self, step, extra_info=None, yolo_data=None):
        """更新Pygame显示"""
        self.env.hud.tick(self.env, self.env.clock)
        
        if step == 0:
            self.env.display.fill((0, 0, 0))
        
        self.env.hud.render(self.env.display)
        
        # 绘制检测视角 (画中画)
        if yolo_data:
            img_rgb, tl_results = yolo_data
            if img_rgb is not None:
                # 绘制 YOLO 边框 (UFLD 车道线已经绘制在 img_rgb 上了)
                if tl_results:
                    for bbox, info in tl_results.items():
                        x1, y1, x2, y2 = bbox
                        color_name = info['color']
                        conf = info.get('conf', 0.0)
                        
                        if color_name == 'RED':
                            box_color = (255, 0, 0)
                        elif color_name == 'YELLOW':
                            box_color = (255, 255, 0)
                        elif color_name == 'GREEN':
                            box_color = (0, 255, 0)
                        else:
                            box_color = (200, 200, 200)
                        
                        # ROI 检查 (可视化用)
                        img_w = img_rgb.shape[1]
                        center_x = (x1 + x2) / 2
                        is_in_roi = (img_w * 0.25 < center_x < img_w * 0.75)
                        
                        thickness = 2 if is_in_roi else 1
                        line_type = cv2.LINE_AA if is_in_roi else cv2.LINE_4
                        
                        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), box_color, thickness, line_type)
                        
                        status_text = "" if is_in_roi else "(IGNORED)"
                        label = f"{color_name} {conf:.2f} {status_text}"
                        cv2.putText(img_rgb, label, (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)
                
                # 绘制 ROI 辅助线 (两条白色竖线)
                img_h, img_w, _ = img_rgb.shape
                cv2.line(img_rgb, (int(img_w * 0.25), 0), (int(img_w * 0.25), img_h), (255, 255, 255), 1)
                cv2.line(img_rgb, (int(img_w * 0.75), 0), (int(img_w * 0.75), img_h), (255, 255, 255), 1)

                # 将图像转为 Pygame surface 并显示
                yolo_surface = pygame.surfarray.make_surface(img_rgb.swapaxes(0, 1))
                self.env.display.blit(yolo_surface, (640, 0))
                pygame.draw.rect(self.env.display, (255, 255, 255), (640, 0, 640, 480), 2)
        
        # 显示警告文字
        if extra_info:
            font = pygame.font.SysFont("Arial", 30, bold=True)
            text_surface = font.render(extra_info, True, (255, 0, 0))
            text_rect = text_surface.get_rect(center=(640, 500))
            self.env.display.blit(text_surface, text_rect)

        pygame.display.flip()
        self.env.check_quit()
    
    def _check_destination_reached(self, next_state):
        """检查是否到达目的地"""
        x, y = next_state[0][0], next_state[0][1]
        end_x, end_y = self.agent._end_transform.location.x, self.agent._end_transform.location.y
        
        distance = np.linalg.norm([x - end_x, y - end_y])
        return distance < self.simulation_params['destination_threshold']
    
    def _record_simulation_data(self, step, next_state, acceleration, steering):
        """记录仿真数据"""
        x, y, _, vx, _, _ = next_state[0]
        current_time = step * self.simulation_params['time_step']
        
        self.simulation_data['trajectory'].append([x, y])
        self.simulation_data['velocities'].append(vx)
        self.simulation_data['accelerations'].append(acceleration)
        self.simulation_data['steerings'].append(steering)
        self.simulation_data['times'].append(current_time)
    
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

    def _process_yolo_detection(self):
        """处理图像检测：YOLO (红绿灯) + UFLD (车道线) + CV (停止线)"""
        is_red_light = False
        info_text = ""
        img_rgb = None
        tl_results = {}
        stop_line_detected = False
        
        # 清空队列，只取最新一帧
        image_data = None
        while not self.rgb_queue.empty():
            image_data = self.rgb_queue.get_nowait()
        
        if image_data is None:
            return is_red_light, info_text, img_rgb, tl_results, stop_line_detected

        try:
            array = np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image_data.height, image_data.width, 4))
            raw_img_rgb = array[:, :, :3][:, :, ::-1].copy()
            img_rgb = raw_img_rgb.copy()
            
            # 1. UFLD 车道线检测
            if self.lane_detector:
                img_rgb = self.lane_detector.detect_lanes(raw_img_rgb)
            
            # 2. 停止线检测 (辅助)
            stop_line_detected = detect_stop_lines(raw_img_rgb)
            if stop_line_detected:
                # 绘制停止线提示
                cv2.putText(img_rgb, "STOP LINE", (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # 3. YOLO 红绿灯检测
            if self.yolo_model:
                results = self.yolo_model(raw_img_rgb, verbose=False)
                
                tl_results = traffic_light.detect_traffic_lights(
                    raw_img_rgb, 
                    results, 
                    self.yolo_model.names, 
                    method='hsv'
                )
                
                img_w = img_rgb.shape[1]
                
                for bbox, info in tl_results.items():
                    color = info['color']
                    
                    if color == 'RED':
                        x1, y1, x2, y2 = bbox
                        center_x = (x1 + x2) / 2
                        height = y2 - y1
                        
                        # ROI 区域过滤
                        if center_x < img_w * 0.25 or center_x > img_w * 0.75:
                            print(f"Ignoring RED light at x={center_x:.1f} (outside ROI)")
                            continue
                        
                        if height > 20: 
                            is_red_light = True
                            info_text = "RED LIGHT! STOP!"
                            print(f"Red light detected! Box height: {height}, Center X: {center_x:.1f}")
                            break
                            
        except Exception as e:
            print(f"Detection processing error: {e}")
            import traceback
            traceback.print_exc()
                
        return is_red_light, info_text, img_rgb, tl_results, stop_line_detected

    def _is_vehicle_in_junction(self):
        """检查车辆是否位于路口内部"""
        if self.env.ego_vehicle and self.env.map:
            loc = self.env.ego_vehicle.get_location()
            waypoint = self.env.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            return waypoint.is_junction
        return False
    
    def run_simulation(self, start_idx=75, end_idx=40):
        """运行完整的仿真流程"""
        try:
            self._setup_route(start_idx, end_idx)
            self._initialize_vehicle_and_agent(start_idx, end_idx)
            
            if self.env.display_method == "pygame":
                self.env.init_display()
            
            self._setup_yolo_sensor()
            
            for step in range(self.simulation_params['max_simulation_steps']):
                try:
                    # 1. 运行检测 (YOLO + UFLD + StopLine)
                    is_red_light, tl_info, img_rgb, tl_results, is_stop_line = self._process_yolo_detection()

                    # 2. 执行 MPC 控制
                    acceleration_opt, steering_opt, next_state = self.agent.run_step()
                    
                    # 3. 红绿灯停车逻辑覆盖
                    # 红灯 且 (未在路口内 或 检测到停止线) 时停车
                    if is_red_light and not self._is_vehicle_in_junction():
                        acceleration_opt = -4.0
                        if is_red_light:
                            tl_info = "RED LIGHT! STOP!"
                            if is_stop_line:
                                tl_info += " (STOP LINE)"
                    
                    self._record_simulation_data(
                        step, next_state, acceleration_opt, steering_opt
                    )
                    
                    self.env.step([acceleration_opt, steering_opt])
                    
                    if self.env.display_method == "pygame":
                        yolo_data = (img_rgb, tl_results) if img_rgb is not None else None
                        self._update_pygame_display(step, extra_info=tl_info, yolo_data=yolo_data)
                    
                    if self._check_destination_reached(next_state):
                        print("Destination reached!")
                        if self.env.display_method == "pygame":
                            pygame.quit()
                        break
                    
                except Exception as e:
                    print(f"Simulation step error: {e}")
                    continue
            
        except KeyboardInterrupt:
            print("Simulation interrupted by user.")
        except Exception as e:
            print(f"Critical simulation error: {e}")
        finally:
            self._visualize_results()


def main():
    simulation = MPCCarSimulation()
    simulation.run_simulation()


if __name__ == "__main__":
    main()