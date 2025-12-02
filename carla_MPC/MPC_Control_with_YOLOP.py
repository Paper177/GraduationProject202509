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

# --- 辅助函数：停止线检测 (OpenCV) ---
def detect_stop_lines(img_rgb):
    """检测路口停止线 (保留 OpenCV 逻辑作为辅助)"""
    height, width = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    mask = np.zeros_like(edges)
    # 梯形 ROI
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

# --- YOLOP 检测类 ---
class YOLOPDetector:
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = None
        
        if not TORCH_AVAILABLE:
            return

        # 1. 尝试定位本地 YOLOP 仓库
        yolop_local_path = pathlib.Path('carla_MPC/YOLOP')
        
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
        """
        if self.model is None:
            return img_rgb

        img_h, img_w = img_rgb.shape[:2]
        
        # 1. 预处理
        # YOLOP 需要 resize 到 640x640
        img_resized = cv2.resize(img_rgb, (640, 640))
        input_tensor = self.transform(img_resized).unsqueeze(0).to(self.device)
        
        # 2. 推理
        with torch.no_grad():
            # det_out: (1, n, 6) detection
            # da_seg_out: (1, 2, 640, 640) drivable area
            # ll_seg_out: (1, 2, 640, 640) lane line
            det_out, da_seg_out, ll_seg_out = self.model(input_tensor)
            
        # 3. 后处理 - 分割掩码
        # Drivable Area
        da_seg_mask = torch.nn.functional.interpolate(da_seg_out, size=(img_h, img_w), mode='bilinear', align_corners=True)
        da_seg_mask = torch.argmax(da_seg_mask, dim=1).squeeze().cpu().numpy() # 0 or 1
        
        # Lane Line
        ll_seg_mask = torch.nn.functional.interpolate(ll_seg_out, size=(img_h, img_w), mode='bilinear', align_corners=True)
        ll_seg_mask = torch.argmax(ll_seg_mask, dim=1).squeeze().cpu().numpy() # 0 or 1
        
        # 4. 可视化叠加
        img_vis = img_rgb.copy()
        
        # 绘制可行驶区域 (蓝色，半透明)
        color_area = np.zeros_like(img_vis)
        color_area[da_seg_mask == 1] = [0, 0, 255] # RGB: Blue
        # 只有掩码区域才混合
        mask_bool = (da_seg_mask == 1)
        img_vis[mask_bool] = cv2.addWeighted(img_vis[mask_bool], 0.7, color_area[mask_bool], 0.3, 0).squeeze()
        
        # 绘制车道线 (绿色，不透明或高亮)
        # 膨胀一下车道线让显示更清晰
        ll_seg_mask = (ll_seg_mask * 255).astype(np.uint8)
        # kernel = np.ones((3,3), np.uint8)
        # ll_seg_mask = cv2.dilate(ll_seg_mask, kernel, iterations=1)
        
        img_vis[ll_seg_mask > 100] = [0, 255, 0] # RGB: Green
        
        return img_vis


class MPCCarSimulation:
    """MPC车辆控制仿真主类 (集成 YOLOP 车道/路面检测 + YOLOv11 红绿灯检测)"""
    
    def __init__(self):
        """初始化仿真参数和环境"""
        # 仿真参数配置
        self.simulation_params = {
            'time_step': 0.05,         # 仿真时间步长（秒）
            'target_speed': 30,        # 目标速度（km/h）
            'sample_resolution': 2.0,  # 路径规划采样分辨率
            'display_mode': "pygame",  # 显示模式
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
        self.yolo_model = None      # YOLOv11 (红绿灯)
        self.yolop_detector = None  # YOLOP (车道线/路面)
        self.rgb_queue = queue.Queue()
        self.yolo_sensor = None
        
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
    
    def _setup_yolo_sensor(self):
        """设置车顶 RGB 摄像头"""
        if self.vehicle is None: return

        bp_library = self.env.world.get_blueprint_library()
        camera_bp = bp_library.find('sensor.camera.rgb')
        # 640x480 是兼顾 YOLOP (640x640) 和 YOLOv11 的合理分辨率
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '45') # 较窄 FOV 适合看红绿灯
        
        spawn_point = carla.Transform(carla.Location(x=1.0, z=2.0))
        
        self.yolo_sensor = self.env.world.spawn_actor(
            camera_bp, spawn_point, attach_to=self.env.ego_vehicle
        )
        self.yolo_sensor.listen(self.rgb_queue.put)
        self.env.actor_list.append(self.yolo_sensor)
        print("RGB Sensor initialized.")

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
    
    def _update_pygame_display(self, step, extra_info=None, yolo_data=None):
        """更新显示"""
        self.env.hud.tick(self.env, self.env.clock)
        if step == 0: self.env.display.fill((0, 0, 0))
        self.env.hud.render(self.env.display)
        
        # 画中画显示
        if yolo_data:
            img_rgb, tl_results = yolo_data
            if img_rgb is not None:
                # 绘制红绿灯框 (img_rgb 已经包含了 YOLOP 的车道线渲染)
                for bbox, info in tl_results.items():
                    x1, y1, x2, y2 = bbox
                    color_name = info['color']
                    conf = info.get('conf', 0.0)
                    
                    colors = {'RED': (255, 0, 0), 'YELLOW': (255, 255, 0), 'GREEN': (0, 255, 0)}
                    box_color = colors.get(color_name, (200, 200, 200))
                    
                    # ROI Check
                    img_w = img_rgb.shape[1]
                    center_x = (x1 + x2) / 2
                    is_in_roi = (img_w * 0.25 < center_x < img_w * 0.75)
                    
                    thickness = 2 if is_in_roi else 1
                    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), box_color, thickness)
                    
                    status = "" if is_in_roi else "(IGN)"
                    label = f"{color_name} {conf:.2f} {status}"
                    cv2.putText(img_rgb, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)

                # ROI Lines
                h, w, _ = img_rgb.shape
                cv2.line(img_rgb, (int(w*0.25), 0), (int(w*0.25), h), (255,255,255), 1)
                cv2.line(img_rgb, (int(w*0.75), 0), (int(w*0.75), h), (255,255,255), 1)

                surf = pygame.surfarray.make_surface(img_rgb.swapaxes(0, 1))
                self.env.display.blit(surf, (640, 0))
                pygame.draw.rect(self.env.display, (255, 255, 255), (640, 0, 640, 480), 2)
        
        if extra_info:
            font = pygame.font.SysFont("Arial", 30, bold=True)
            text = font.render(extra_info, True, (255, 0, 0))
            self.env.display.blit(text, text.get_rect(center=(640, 500)))

        pygame.display.flip()
        self.env.check_quit()

    def _process_yolo_detection(self):
        """
        处理流程：
        1. 获取最新图像
        2. YOLOP -> 检测车道线 & 可行驶区域 -> 绘制到图像
        3. OpenCV -> 检测停止线 -> 绘制到图像
        4. YOLOv11 -> 检测红绿灯 -> 返回结果
        """
        is_red_light = False
        info_text = ""
        img_rgb = None
        tl_results = {}
        stop_line_detected = False
        
        # 1. 队列处理
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
            
            # 2. YOLOP 推理 (Lane + Drivable Area)
            if self.yolop_detector:
                img_rgb = self.yolop_detector.infer(raw_img_rgb)
            
            # 3. 停止线检测 (辅助)
            stop_line_detected = detect_stop_lines(raw_img_rgb)
            if stop_line_detected:
                cv2.putText(img_rgb, "STOP LINE", (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # 4. YOLOv11 红绿灯检测
            if self.yolo_model:
                results = self.yolo_model(raw_img_rgb, verbose=False)
                tl_results = traffic_light.detect_traffic_lights(
                    raw_img_rgb, results, self.yolo_model.names, method='hsv'
                )
                
                img_w = img_rgb.shape[1]
                for bbox, info in tl_results.items():
                    if info['color'] == 'RED':
                        x1, y1, x2, y2 = bbox
                        center_x = (x1 + x2) / 2
                        height = y2 - y1
                        
                        # ROI 过滤 (只看中间 50%)
                        if center_x < img_w * 0.25 or center_x > img_w * 0.75:
                            continue
                        
                        if height > 20: 
                            is_red_light = True
                            info_text = "RED LIGHT! STOP!"
                            break
                            
        except Exception as e:
            print(f"Detection error: {e}")
            import traceback
            traceback.print_exc()
            
        return is_red_light, info_text, img_rgb, tl_results, stop_line_detected

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
    
    def run_simulation(self, start_idx=75, end_idx=40):
        try:
            self._setup_route(start_idx, end_idx)
            self._initialize_vehicle_and_agent(start_idx, end_idx)
            if self.env.display_method == "pygame": self.env.init_display()
            self._setup_yolo_sensor()
            
            for step in range(self.simulation_params['max_simulation_steps']):
                is_red, info, img, tl_res, is_stop = self._process_yolo_detection()
                
                acc, steer, next_state = self.agent.run_step()
                
                # 停车逻辑: 红灯 且 (不在路口 或 看到停止线)
                if is_red and not self._is_vehicle_in_junction():
                    acc = -4.0
                    if is_stop: info += " (LINE)"
                
                self._record_simulation_data(step, next_state, acc, steer)
                self.env.step([acc, steer])
                
                if self.env.display_method == "pygame":
                    yolo_data = (img, tl_res) if img is not None else None
                    self._update_pygame_display(step, extra_info=info, yolo_data=yolo_data)
                
                if self._check_destination_reached(next_state):
                    print("Destination reached!"); break
                    
        except KeyboardInterrupt:
            print("User interrupted.")
        except Exception as e:
            print(f"Simulation error: {e}")
        finally:
            self._visualize_results()

def main():
    sim = MPCCarSimulation()
    sim.run_simulation()

if __name__ == "__main__":
    main()