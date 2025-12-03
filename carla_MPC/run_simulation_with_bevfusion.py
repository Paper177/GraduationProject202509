import numpy as np
import matplotlib.pyplot as plt
from src.mcp_controller import Vehicle
from env import Env, draw_waypoints
import carla
import cv2
import queue
import copy
import sys, pathlib
import torch  # 需要安装 torch
from collections import deque

sys.path.insert(0, str(pathlib.Path(__file__).with_name('src')))
from src.x_v2x_agent import Xagent
from src.global_route_planner import GlobalRoutePlanner

import time
import pygame

# ==========================================
# 1. BEVFusion 传感器管理器
# ==========================================
class SensorManager:
    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.sensors = {}
        self.queues = {}
        self.data_dict = {}
        
        # BEVFusion / nuScenes 典型配置
        # 6个相机 (Front, Front-Right, Front-Left, Back, Back-Right, Back-Left) + 1个 LiDAR
        self.camera_transforms = {
            'CAM_FRONT': carla.Transform(carla.Location(x=1.5, z=1.6), carla.Rotation(pitch=0, yaw=0)),
            'CAM_FRONT_RIGHT': carla.Transform(carla.Location(x=1.5, y=0.5, z=1.6), carla.Rotation(pitch=0, yaw=55)),
            'CAM_FRONT_LEFT': carla.Transform(carla.Location(x=1.5, y=-0.5, z=1.6), carla.Rotation(pitch=0, yaw=-55)),
            'CAM_BACK': carla.Transform(carla.Location(x=-1.5, z=1.6), carla.Rotation(pitch=0, yaw=180)),
            'CAM_BACK_RIGHT': carla.Transform(carla.Location(x=-1.5, y=0.5, z=1.6), carla.Rotation(pitch=0, yaw=110)),
            'CAM_BACK_LEFT': carla.Transform(carla.Location(x=-1.5, y=-0.5, z=1.6), carla.Rotation(pitch=0, yaw=-110)),
        }
        self.lidar_transform = carla.Transform(carla.Location(x=0.0, z=1.8), carla.Rotation(pitch=0, yaw=0))

        self._setup_sensors()

    def _setup_sensors(self):
        bp_library = self.world.get_blueprint_library()

        # 1. Setup LiDAR
        lidar_bp = bp_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('range', '100')
        lidar_bp.set_attribute('rotation_frequency', '20')
        lidar_bp.set_attribute('points_per_second', '320000')
        
        self.sensors['LIDAR_TOP'] = self.world.spawn_actor(
            lidar_bp, self.lidar_transform, attach_to=self.vehicle)
        self.queues['LIDAR_TOP'] = queue.Queue()
        self.sensors['LIDAR_TOP'].listen(self.queues['LIDAR_TOP'].put)

        # 2. Setup Cameras
        cam_bp = bp_library.find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '800')
        cam_bp.set_attribute('image_size_y', '450')
        cam_bp.set_attribute('fov', '70')
        
        for name, transform in self.camera_transforms.items():
            cam = self.world.spawn_actor(cam_bp, transform, attach_to=self.vehicle)
            self.sensors[name] = cam
            self.queues[name] = queue.Queue()
            cam.listen(self.queues[name].put)

    def get_sensor_data(self):
        """同步获取所有传感器数据"""
        data = {}
        try:
            # 简单同步：假设帧率足够高，取最新一帧
            for name, q in self.queues.items():
                data[name] = q.get(timeout=2.0)
        except queue.Empty:
            print("Warning: Sensor data missing/timeout")
            return None
        return data

    def destroy(self):
        for s in self.sensors.values():
            s.stop()
            s.destroy()

# ==========================================
# 2. BEVFusion 感知接口 (Mock/Wrapper)
# ==========================================
class BEVFusionPerception:
    def __init__(self, model_path=None):
        """
        初始化 BEVFusion 模型
        实际使用时，你需要在这里加载 PyTorch 模型
        """
        print("Initializing BEVFusion Model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # TODO: 如果你有实际的 bevfusion 权重和代码库，在这里加载
        # self.model = build_model(config).to(self.device)
        # self.model.load_state_dict(torch.load(model_path))
        
        # 相机内参 (基于 CARLA 设置: 800x450, FOV 70)
        self.intrinsics = self._get_camera_intrinsics(800, 450, 70)

    def _get_camera_intrinsics(self, w, h, fov):
        f = w / (2.0 * np.tan(fov * np.pi / 360.0))
        K = np.identity(3)
        K[0, 0] = K[1, 1] = f
        K[0, 2] = w / 2.0
        K[1, 2] = h / 2.0
        return K

    def preprocess(self, sensor_data, ego_transform):
        """
        将 CARLA 数据转换为 BEVFusion 输入格式
        """
        if sensor_data is None: return None
        
        inputs = {
            'img': [],
            'points': [],
            'camera_intrinsics': [],
            'camera2ego': [],
            'lidar2ego': [],
            'lidar2camera': []
        }

        # 处理 LiDAR (x, y, z, intensity)
        lidar_data = sensor_data['LIDAR_TOP']
        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        # BEVFusion 通常需要 [x, y, z, intensity, ring_index(optional)]
        # 这里为了演示，我们只保留 numpy array
        inputs['points'] = points 

        # 处理相机数据
        # BEVFusion 需要构建 3D 坐标系转换矩阵
        for name in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
                     'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT']:
            
            # 图像转换
            carla_img = sensor_data[name]
            img_array = np.frombuffer(carla_img.raw_data, dtype=np.dtype("uint8"))
            img_array = np.reshape(img_array, (carla_img.height, carla_img.width, 4))
            img_array = img_array[:, :, :3][:, :, ::-1] # BGRA -> RGB
            
            # 将图像标准化并转为 Tensor (N, C, H, W)
            img_tensor = torch.from_numpy(img_array.copy()).permute(2, 0, 1).float()
            inputs['img'].append(img_tensor)

            # 内参
            inputs['camera_intrinsics'].append(torch.from_numpy(self.intrinsics).float())
            
            # 外参 (简化处理：这里应该计算相对 ego 的变换矩阵)
            # 在实际 BEVFusion 中，你需要计算 camera2lidar 或 camera2ego 的 4x4 矩阵
            # 这里仅作占位符
            inputs['camera2ego'].append(torch.eye(4)) 

        # Stack Tensors
        inputs['img'] = torch.stack(inputs['img']).to(self.device)
        
        return inputs

    def inference(self, sensor_data, ego_transform):
        """
        运行感知推理
        """
        # 1. 预处理
        input_dict = self.preprocess(sensor_data, ego_transform)
        if input_dict is None: return []

        # 2. 模型前向传播 (Mock)
        # with torch.no_grad():
        #    results = self.model(return_loss=False, **input_dict)
        
        # 3. 后处理 (Mock 输出)
        # 假设检测到了前方 10 米处的一个物体
        # 格式: [x, y, z, dx, dy, dz, yaw, velocity_x, velocity_y]
        
        mock_detection = []
        # 示例：每 50 帧模拟检测到一个障碍物
        if int(time.time() * 10) % 50 == 0:
            print("BEVFusion: Detected Obstacle at (10.0, 0.0)")
            mock_detection.append({
                'box3d': np.array([10.0, 0.0, 0.0, 2.0, 4.0, 1.5, 0.0]),
                'score': 0.95,
                'label': 1 # Vehicle
            })
            
        return mock_detection

# ==========================================
# 3. 主程序
# ==========================================
def main():
    # Simulation parameters
    simu_step = 0.05
    target_v = 40
    sample_res = 2.0
    display_mode = "pygame" # 强制使用 pygame 以便看到输出

    env = Env(display_method=display_mode, dt=simu_step)
    env.clean() # 清理残留 actor
    spawn_points = env.map.get_spawn_points()

    start_idx, end_idx = 87, 40

    # 1. 设置路径
    grp = GlobalRoutePlanner(env.map, sample_res)
    route = grp.trace_route(spawn_points[start_idx].location, spawn_points[end_idx].location)
    draw_waypoints(env.world, [wp for wp, _ in route], z=0.5, color=(0, 255, 0))
    
    # 2. 生成车辆
    env.reset(spawn_point=spawn_points[start_idx])

    # 3. 初始化 MPC 智能体
    dynamic_model = Vehicle(actor=env.ego_vehicle, horizon=10, target_v=target_v, delta_t=simu_step, max_iter=30)
    agent = Xagent(env, dynamic_model, dt=simu_step)
    agent.set_start_end_transforms(start_idx, end_idx)
    agent.plan_route(agent._start_transform, agent._end_transform)

    # 4. 初始化 BEVFusion 模块
    sensor_manager = SensorManager(env.world, env.ego_vehicle)
    perception_module = BEVFusionPerception()

    print("Sensors initialized. Starting simulation...")

    trajectory = []
    velocities = []
    accelerations = []
    steerings = []
    times = []

    if env.display_method == "pygame":
        env.init_display()

    try:
        max_sim_steps = 2000
        for step in range(max_sim_steps):
            
            # --- Perception Step ---
            # 获取传感器数据
            sensor_data = sensor_manager.get_sensor_data()
            ego_transform = env.ego_vehicle.get_transform()
            
            # 运行 BEVFusion 推理
            detections = perception_module.inference(sensor_data, ego_transform)
            
            # 如果检测到障碍物，简单的紧急制动逻辑 (示例)
            obstacle_detected = False
            for obj in detections:
                # 简单逻辑：如果前方 x < 15m 有物体
                if 0 < obj['box3d'][0] < 15.0 and abs(obj['box3d'][1]) < 2.0:
                    obstacle_detected = True
                    print(f"!!! Emergency Brake: Obstacle at {obj['box3d'][0]:.1f}m !!!")
            
            # --- Control Step ---
            if obstacle_detected:
                a_opt = -4.0 # 紧急刹车
                delta_opt = 0.0
                # 获取当前状态仅用于记录
                state, _ = dynamic_model.get_state_carla()
                next_state = [[state[0], state[1], state[2], state[3], state[4], state[5]]] # Dummy
            else:
                # 正常 MPC 控制
                try:
                    a_opt, delta_opt, next_state = agent.run_step()
                except Exception as e:
                    print(f"MPC Error: {e}")
                    a_opt, delta_opt = 0.0, 0.0
                    next_state = [[0,0,0,0,0,0]]

            # 记录数据
            x, y, yaw, vx, vy, omega = next_state[0]
            trajectory.append([x, y]) 
            velocities.append(vx)  
            accelerations.append(a_opt)  
            steerings.append(delta_opt)  
            times.append(step * simu_step)  
            
            # 执行控制
            env.step([a_opt, delta_opt])
            
            # Visualization
            if env.display_method == "pygame":
                env.hud.tick(env, env.clock)
                if step == 0:  
                    env.display.fill((0, 0, 0))  
                env.hud.render(env.display)  
                
                # 可选：在 Pygame 窗口上绘制 BEVFusion 的检测框（需要将 3D 框投影到 2D）
                # 这里略过复杂的投影代码
                
                pygame.display.flip()        
                env.check_quit()

            # 检查到达终点
            if np.linalg.norm([env.ego_vehicle.get_location().x - agent._end_transform.location.x,
                               env.ego_vehicle.get_location().y - agent._end_transform.location.y]) < 2.0:
                print("Destination reached!")
                break 

            if env.display_method == "pygame":
                # 移除 time.sleep，因为 sensor queue.get 已经起到了同步作用，或者保持较小的 sleep
                pass 

    except KeyboardInterrupt:
        print("Simulation interrupted by user.")
    finally:
        # 清理
        print("Cleaning up sensors and actors...")
        sensor_manager.destroy()
        if env.display_method == "pygame":
            pygame.quit()
        env.clean()

    # 绘图代码保持不变...
    trajectory = np.array(trajectory)
    # (省略绘图部分，与原文件一致)
    plt.close('all') # 避免阻塞

if __name__ == '__main__':
    main()