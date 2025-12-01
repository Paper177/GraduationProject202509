import numpy as np
import matplotlib.pyplot as plt
from src.mcp_controller import Vehicle
from env import Env, draw_waypoints

import sys
import pathlib
import queue
import cv2

# --- 添加路径以导入 Traffic_detection 模块 ---
traffic_detection_path = pathlib.Path(__file__).parent.parent / 'Traffic_detection'
sys.path.insert(0, str(traffic_detection_path))
sys.path.insert(0, str(pathlib.Path(__file__).with_name('src')))

# 尝试导入 YOLO 和 traffic_light
try:
    from ultralytics import YOLO
    import traffic_light
    YOLO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: YOLO or traffic_light module not found. Traffic light detection disabled. {e}")
    YOLO_AVAILABLE = False

from src.x_v2x_agent import Xagent
from src.global_route_planner import GlobalRoutePlanner

import time
import pygame
import carla


class MPCCarSimulation:
    """MPC车辆控制仿真主类 (集成YOLO红绿灯检测与可视化)"""
    
    def __init__(self):
        """初始化仿真参数和环境"""
        # 仿真参数配置
        self.simulation_params = {
            'time_step': 0.05,  # 仿真时间步长（秒）
            'target_speed': 60,  # 目标速度（km/h）
            'sample_resolution': 2.0,  # 路径规划采样分辨率
            'display_mode': "pygame",  # 显示模式："spec" 或 "pygame"
            'max_simulation_steps': 5000,  # 最大仿真步数
            'destination_threshold': 1.0  # 到达目的地的距离阈值（米）
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
        
        # YOLO 相关
        self.yolo_model = None
        self.rgb_queue = queue.Queue()
        self.yolo_sensor = None
        self._init_yolo_model()

    def _init_yolo_model(self):
        """初始化 YOLO 模型"""
        if not YOLO_AVAILABLE:
            return
            
        model_path = pathlib.Path("YoloModel/yolo11x.pt")
        try:
            print(f"Loading YOLO model from {model_path}...")
            # 如果本地有模型则加载，否则会自动下载
            self.yolo_model = YOLO(str(model_path)) if model_path.exists() else YOLO("yolo11s.pt")
            print("YOLO model loaded successfully.")
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            self.yolo_model = None

    def _initialize_environment(self):
        """初始化仿真环境，并确保加载 Town05 地图"""
        host = 'localhost'
        port = 2000
        
        # 1. 检查并加载 Town05
        try:
            client = carla.Client(host, port)
            client.set_timeout(10.0)
            world = client.get_world()
            current_map_name = world.get_map().name
            
            # CARLA 地图名称通常包含路径，如 "/Game/Carla/Maps/Town05"
            if 'Town05' not in current_map_name:
                print(f"Current map is {current_map_name}. Loading Town05...")
                client.load_world('Town05')
                print("Town05 loaded successfully.")
            else:
                print("Town05 is already loaded.")
                
        except Exception as e:
            print(f"Error checking/loading map: {e}")

        # 2. 初始化 Env 类
        # 注意：Env类内部会再次获取 world，所以这里不需要传递 world 对象
        env = Env(
            host=host,
            port=port,
            display_method=self.simulation_params['display_mode'],
            dt=self.simulation_params['time_step']
        )
        env.clean()
        return env
    
    def _setup_yolo_sensor(self):
        """设置用于 YOLO 检测的 RGB 摄像头"""
        if self.vehicle is None or self.yolo_model is None:
            return

        bp_library = self.env.world.get_blueprint_library()
        camera_bp = bp_library.find('sensor.camera.rgb')
        # 设置适中的分辨率以平衡性能和检测精度
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        # FOV 设置为 45 度，以便更清楚地看到远处的红绿灯
        camera_bp.set_attribute('fov', '45')
        
        # 安装在车顶位置 (第一人称视角)
        spawn_point = carla.Transform(carla.Location(x=1.0, z=2.0))
        
        self.yolo_sensor = self.env.world.spawn_actor(
            camera_bp, 
            spawn_point, 
            attach_to=self.env.ego_vehicle
        )
        self.yolo_sensor.listen(self.rgb_queue.put)
        # 将传感器加入 env 的 actor 列表以便统一清理
        self.env.actor_list.append(self.yolo_sensor)
        print("YOLO sensor initialized.")

    def _setup_route(self, start_idx, end_idx):
        """设置起点和终点，规划全局路径"""
        # 创建全局路径规划器
        route_planner = GlobalRoutePlanner(
            self.env.map, 
            self.simulation_params['sample_resolution']
        )
        
        # 规划路径
        self.route = route_planner.trace_route(
            self.spawn_points[start_idx].location, 
            self.spawn_points[end_idx].location
        )
        
        # 可视化路径点
        draw_waypoints(
            self.env.world, 
            [wp for wp, _ in self.route], 
            z=0.5, 
            color=(0, 255, 0)
        )
        
        # 重置环境到起点
        self.env.reset(spawn_point=self.spawn_points[start_idx])
        
    def _initialize_vehicle_and_agent(self, start_idx, end_idx):
        """初始化车辆动力学模型和智能体"""
        # 初始化车辆动力学模型
        self.vehicle = Vehicle(
            actor=self.env.ego_vehicle,
            horizon=10,
            target_v=self.simulation_params['target_speed'],
            delta_t=self.simulation_params['time_step'],
            max_iter=30
        )
        
        # 初始化智能体
        self.agent = Xagent(
            self.env, 
            self.vehicle, 
            dt=self.simulation_params['time_step']
        )
        self.agent.set_start_end_transforms(start_idx, end_idx)
        self.agent.plan_route(self.agent._start_transform, self.agent._end_transform)
    
    def _update_pygame_display(self, step, extra_info=None, yolo_data=None):
        """更新Pygame显示
        Args:
            step: 当前步数
            extra_info: 文本提示信息
            yolo_data: (img_rgb, tl_results) 元组，包含图像和检测结果
        """
        # 1. 更新 HUD 内容
        self.env.hud.tick(self.env, self.env.clock)
        
        # 2. 如果是第一帧，填充背景
        if step == 0:
            self.env.display.fill((0, 0, 0))
        
        # 3. 渲染 HUD (这会绘制主摄像头背景和左侧文字信息)
        # 注意：self.env.hud.render 内部并未绘制 camera 图像，
        # camera 图像是由 env.py 中的 camera_callback 异步 blit 到 self.env.display 的。
        # 这里我们假设主画面已经由回调函数更新了。
        self.env.hud.render(self.env.display)
        
        # 4. 绘制 YOLO 检测视角 (画中画)
        if yolo_data:
            img_rgb, tl_results = yolo_data
            if img_rgb is not None:
                # 在图像上绘制检测框
                # img_rgb 是 RGB 格式, cv2 默认是 BGR, 但我们这里只用来绘图，
                # pygame 需要 RGB，所以我们直接用 RGB 画，颜色参数传 RGB 即可。
                
                for bbox, info in tl_results.items():
                    x1, y1, x2, y2 = bbox
                    color_name = info['color']
                    conf = info.get('conf', 0.0)
                    
                    # 定义绘制颜色 (RGB)
                    if color_name == 'RED':
                        box_color = (255, 0, 0)
                    elif color_name == 'YELLOW':
                        box_color = (255, 255, 0)
                    elif color_name == 'GREEN':
                        box_color = (0, 255, 0)
                    else:
                        box_color = (200, 200, 200) # 未知/灰色
                    
                    # 画框
                    cv2.rectangle(img_rgb, (x1, y1), (x2, y2), box_color, 2)
                    # 画标签
                    label = f"{color_name} {conf:.2f}"
                    cv2.putText(img_rgb, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

                # 将 numpy 数组转换为 pygame surface
                # img_rgb shape: (480, 640, 3) -> (Width, Height) for Pygame
                # Pygame make_surface 需要 (Width, Height, Depth), numpy 是 (Row, Col, Depth) 即 (H, W, D)
                # 所以需要 swapaxes
                yolo_surface = pygame.surfarray.make_surface(img_rgb.swapaxes(0, 1))
                
                # 绘制到主屏幕右上角
                # 主屏幕默认 1280x720, YOLO 640x480
                # 放置位置: x = 1280 - 640 = 640, y = 0
                self.env.display.blit(yolo_surface, (640, 0))
                
                # 画一个边框区分两个视图
                pygame.draw.rect(self.env.display, (255, 255, 255), (640, 0, 640, 480), 2)
        
        # 5. 显示额外的警告文字 (如红灯停车)
        if extra_info:
            font = pygame.font.SysFont("Arial", 30, bold=True)
            text_surface = font.render(extra_info, True, (255, 0, 0)) # 红色字体
            # 显示在屏幕中央偏下位置
            text_rect = text_surface.get_rect(center=(640, 500))
            self.env.display.blit(text_surface, text_rect)

        # 6. 刷新屏幕
        pygame.display.flip()
        
        # 7. 检查退出
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
        # 将列表转换为numpy数组
        if not self.simulation_data['times']:
            return

        trajectory = np.array(self.simulation_data['trajectory'])
        velocities = np.array(self.simulation_data['velocities'])
        accelerations = np.array(self.simulation_data['accelerations'])
        steerings = np.array(self.simulation_data['steerings'])
        times = np.array(self.simulation_data['times'])
        
        # 创建2x2子图
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        
        # 1. 车辆轨迹和规划路径
        axs[0, 0].plot(trajectory[:, 0], trajectory[:, 1], 
                      label="Vehicle Path", color='darkorange', linewidth=2)
        axs[0, 0].scatter(self.agent._start_transform.location.x, 
                         self.agent._start_transform.location.y, 
                         color='green', label="Start", zorder=5)
        axs[0, 0].scatter(self.agent._end_transform.location.x, 
                         self.agent._end_transform.location.y, 
                         color='red', label="End", zorder=5)
        
        # 绘制规划路径
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
        
        # 2. 速度-时间曲线
        axs[0, 1].plot(times, velocities, 
                      label="Velocity (m/s)", color='royalblue', linewidth=2)
        axs[0, 1].set_title("Velocity over Time", fontsize=14)
        axs[0, 1].set_xlabel("Time (s)", fontsize=12)
        axs[0, 1].set_ylabel("Velocity (m/s)", fontsize=12)
        axs[0, 1].legend(loc='upper right', fontsize=10)
        axs[0, 1].grid(True)
        
        # 3. 加速度-时间曲线
        axs[1, 0].plot(times, accelerations, 
                      label="Acceleration (m/s²)", color='orange', linewidth=2)
        axs[1, 0].set_title("Acceleration over Time", fontsize=14)
        axs[1, 0].set_xlabel("Time (s)", fontsize=12)
        axs[1, 0].set_ylabel("Acceleration (m/s²)", fontsize=12)
        axs[1, 0].legend(loc='upper right', fontsize=10)
        axs[1, 0].grid(True)
        
        # 4. 转向角-时间曲线
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
        """处理 YOLO 图像检测"""
        is_red_light = False
        info_text = ""
        img_rgb = None
        tl_results = {}
        
        if self.yolo_model:
            # --- 修改开始：清空队列，只取最后一帧 ---
            image_data = None
            # 循环读取队列，丢弃旧帧，直到取到最新的一帧
            while not self.rgb_queue.empty():
                image_data = self.rgb_queue.get_nowait()
            
            # 如果队列原本就是空的（可能是仿真刚开始），则没有数据处理
            if image_data is None:
                return is_red_light, info_text, img_rgb, tl_results
            # --- 修改结束 ---

            try:
                # 后续处理逻辑保持不变
                array = np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (image_data.height, image_data.width, 4))
                
                # 转换为 RGB
                img_rgb = array[:, :, :3][:, :, ::-1].copy()
                
                # ... (其余 YOLO 推理和红绿灯检测代码保持不变) ...
                # array[:, :, :3] 取出 BGR, ::-1 转换为 RGB
                img_rgb = array[:, :, :3][:, :, ::-1].copy()
                
                # YOLO 推理
                results = self.yolo_model(img_rgb, verbose=False)
                
                # 调用 traffic_light 模块进行检测和颜色分析
                # 使用 deep_learning 方法进行二次确认
                tl_results = traffic_light.detect_traffic_lights(
                    img_rgb, 
                    results, 
                    self.yolo_model.names, 
                    method='hsv'
                )
                
                # 分析检测结果
                for bbox, info in tl_results.items():
                    color = info['color']
                    
                    # 简单的过滤逻辑：如果是红灯，且边界框足够大（意味着距离较近）
                    if color == 'RED':
                        # bbox格式: (x1, y1, x2, y2)
                        height = bbox[3] - bbox[1]
                        # 高度阈值，用于忽略远处的红灯，根据实际分辨率(640x480)调整
                        if height > 20: 
                            is_red_light = True
                            info_text = "RED LIGHT! STOP!"
                            print(f"Red light detected! Box height: {height}")
                            break
                            
            except queue.Empty:
                pass
            except Exception as e:
                print(f"YOLO processing error: {e}")
                
        return is_red_light, info_text, img_rgb, tl_results
    
    def run_simulation(self, start_idx=60, end_idx=40):
        """运行完整的仿真流程"""
        try:
            # 设置路径
            self._setup_route(start_idx, end_idx)
            
            # 初始化车辆和智能体
            self._initialize_vehicle_and_agent(start_idx, end_idx)
            
            # 初始化Pygame显示（如果需要）
            if self.env.display_method == "pygame":
                self.env.init_display()
            
            # 初始化 YOLO 传感器
            self._setup_yolo_sensor()
            
            # 主仿真循环
            for step in range(self.simulation_params['max_simulation_steps']):
                try:
                    # 1. 运行 YOLO 检测
                    is_red_light, tl_info, img_rgb, tl_results = self._process_yolo_detection()

                    # 2. 执行 MPC 控制
                    # 首先获取 MPC 建议的控制量
                    acceleration_opt, steering_opt, next_state = self.agent.run_step()
                    
                    # 3. 红绿灯停车逻辑覆盖
                    if is_red_light:
                        acceleration_opt = -4.0  # 强制最大刹车
                        # 保持 MPC 计算的转向，确保车辆在车道内
                    
                    # 记录仿真数据
                    self._record_simulation_data(
                        step, next_state, acceleration_opt, steering_opt
                    )
                    
                    # 执行环境步进
                    self.env.step([acceleration_opt, steering_opt])
                    
                    # 更新Pygame显示
                    if self.env.display_method == "pygame":
                        # 将 YOLO 图像和结果传递给显示函数
                        yolo_data = (img_rgb, tl_results) if img_rgb is not None else None
                        self._update_pygame_display(step, extra_info=tl_info, yolo_data=yolo_data)
                    
                    # 检查是否到达目的地
                    if self._check_destination_reached(next_state):
                        print("Destination reached!")
                        if self.env.display_method == "pygame":
                            pygame.quit()
                        break
                    
                except Exception as e:
                    print(f"Simulation step error: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
        except KeyboardInterrupt:
            print("Simulation interrupted by user.")
        except Exception as e:
            print(f"Critical simulation error: {e}")
        finally:
            # 可视化结果
            self._visualize_results()


def main():
    """主函数"""
    # 创建仿真实例
    simulation = MPCCarSimulation()
    
    # 运行仿真
    simulation.run_simulation()


if __name__ == "__main__":
    main()