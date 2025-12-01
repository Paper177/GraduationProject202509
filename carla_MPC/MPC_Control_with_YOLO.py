import numpy as np
import matplotlib.pyplot as plt
import carla
from src.mcp_controller import Vehicle
from env import Env, draw_waypoints

import sys, pathlib, os

# 添加项目根目录和Traffic_detection目录到系统路径
sys.path.insert(0, str(pathlib.Path(__file__).with_name('src')))
sys.path.insert(0, os.path.abspath(r"e:\GraduationProject202509\Traffic_detection"))

from src.x_v2x_agent import Xagent
from src.global_route_planner import GlobalRoutePlanner

import time
import pygame
import cv2
import queue
from ultralytics import YOLO

# 导入Traffic_detection中的红绿灯识别模块
from traffic_light import (
    get_traffic_light_color,
    detect_traffic_lights,
    YOLO_TARGET_CLASS
)

# YOLO 模型配置
YOLO_MODEL_PATH = r"E:\GraduationProject202509\YoloModel\yolo11x.pt"
YOLO_TARGET_CLASSES = ['traffic light']

# 安全策略参数
SAFE_DIST_TRAFFIC_LIGHT = 30.0  # 红绿灯停车距离 (米)
STOP_DISTANCE = 5.0             # 距离停止线多少米时速度降为0

# 深度图处理函数
def process_depth_image(image, width, height):
    """将 CARLA 深度图转换为米为单位的 2D 数组"""
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (height, width, 4))
    array = array.astype(np.float32)
    normalized_depth = (array[:, :, 2] + array[:, :, 1] * 256 + array[:, :, 0] * 256 * 256) / (256 * 256 * 256 - 1)
    depth_meters = normalized_depth * 1000.0
    return depth_meters

# Simulation parameters
simu_step = 0.05  # Time step per simulation step (seconds)
target_v = 40  # Target speed (km/h)
sample_res = 2.0  # Sampling resolution for path planning
display_mode = "pygame"  # Options: "spec" or "pygame"

# 初始化环境
env = Env(display_method=display_mode, dt=simu_step)
env.clean()

# 设置CARLA地图为Town05（如果需要）
env.world = env.client.load_world('Town05')
env.map = env.world.get_map()

spawn_points = env.map.get_spawn_points()

start_idx, end_idx = 87, 40  # Indices for start and end points

grp = GlobalRoutePlanner(env.map, sample_res)

route = grp.trace_route(spawn_points[start_idx].location, spawn_points[end_idx].location)
draw_waypoints(env.world, [wp for wp, _ in route], z=0.5, color=(0, 255, 0))
env.reset(spawn_point=spawn_points[start_idx])

routes = []
for wp,_ in route:
    wp_t = wp.transform
    routes.append([wp_t.location.x, wp_t.location.y])

# 加载YOLO模型
print("Loading YOLO model...")
try:
    # 检查模型路径是否存在本地文件
    import os
    if os.path.exists(YOLO_MODEL_PATH):
        print(f"从本地文件加载模型: {YOLO_MODEL_PATH}")
        yolo_model = YOLO(YOLO_MODEL_PATH)
    else:
        print(f"尝试从网络下载模型: {YOLO_MODEL_PATH}")
        # 尝试下载模型
        yolo_model = YOLO(YOLO_MODEL_PATH)
    
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"加载 YOLO 模型失败: {e}")
    # 提供更具体的错误信息和建议
    if 'urlopen error' in str(e) or 'timeout' in str(e).lower():
        print("模型下载失败，可能是网络连接问题。请确保您已连接到互联网，或提前下载模型文件并放置在指定路径。")
    print("请检查模型路径是否正确，或尝试使用本地已下载的模型文件。")
    yolo_model = None

# 设置传感器 (RGB + Depth)
blueprint_library = env.world.get_blueprint_library()

# RGB 相机
cam_bp = blueprint_library.find('sensor.camera.rgb')
cam_bp.set_attribute('image_size_x', '1280')
cam_bp.set_attribute('image_size_y', '720')
cam_bp.set_attribute('fov', '30')  # 减小fov以延长检测距离，与Traffic_detection/simulation.py保持一致
cam_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera_rgb = env.world.spawn_actor(cam_bp, cam_transform, attach_to=env.ego_vehicle)

# 深度相机
depth_bp = blueprint_library.find('sensor.camera.depth')
depth_bp.set_attribute('image_size_x', '1280')
depth_bp.set_attribute('image_size_y', '720')
depth_bp.set_attribute('fov', '30')  # 减小fov以延长检测距离，与RGB相机和Traffic_detection/simulation.py保持一致
camera_depth = env.world.spawn_actor(depth_bp, cam_transform, attach_to=env.ego_vehicle)

# 传感器队列
image_queue = queue.Queue()
depth_queue = queue.Queue()
camera_rgb.listen(image_queue.put)
camera_depth.listen(depth_queue.put)

# 初始化 MPC 代理
dynamic_model = Vehicle(actor=env.ego_vehicle, horizon=10, target_v=target_v, delta_t=simu_step, max_iter=30)
agent = Xagent(env, dynamic_model, dt=simu_step)
agent.set_start_end_transforms(start_idx, end_idx)

agent.plan_route(agent._start_transform, agent._end_transform)

sim_time = 0
max_sim_steps = 2000 

trajectory = []
velocities = []
accelerations = []
steerings = []
times = []

# 初始化当前目标速度
current_target_v = target_v

# 用于存储红绿灯检测结果，供pygame显示
tl_detection_results = []

if env.display_method == "pygame":
    env.init_display()

try:
    for step in range(max_sim_steps):
        try:
            # --- A. 获取与处理传感器数据 ---
            img_data = None
            depth_data = None
            img_rgb = None
            depth_map = None
            
            # 尝试获取传感器数据（非阻塞）
            try:
                img_data = image_queue.get(timeout=0.1)
                depth_data = depth_queue.get(timeout=0.1)
                
                # 转换图像格式
                img_array = np.frombuffer(img_data.raw_data, dtype=np.uint8)
                img_array = np.reshape(img_array, (720, 1280, 4))
                img_rgb = img_array[:, :, :3][:, :, ::-1].copy()  # BGR to RGB
                depth_map = process_depth_image(depth_data, 1280, 720)
            except queue.Empty:
                pass  # 没有新数据，继续使用之前的状态
            
            # --- B. 红绿灯检测与决策逻辑 ---
            detected_status = "Cruise"
            
            if yolo_model and img_rgb is not None and depth_map is not None:
                # 使用YOLO track功能检测红绿灯（与Traffic_detection/main.py保持一致）
                results = yolo_model.track(img_rgb, persist=True, verbose=False)
                
                # 使用Traffic_detection中的detect_traffic_lights函数进行红绿灯检测和颜色识别
                tl_results = detect_traffic_lights(img_rgb, results, yolo_model.names, method='hsv')
                
                # 清空之前的检测结果
                tl_detection_results.clear()
                
                # 处理检测到的红绿灯
                for bbox, info in tl_results.items():
                    x1, y1, x2, y2 = bbox
                    tl_color = info['color']
                    conf = info['conf']
                    
                    # 计算红绿灯距离
                    roi_depth = depth_map[y1:y2, x1:x2]
                    if roi_depth.size > 0:
                        dist = np.median(roi_depth[roi_depth > 0]) if np.any(roi_depth > 0) else 0
                    else:
                        dist = 0
                    
                    # 保存检测结果，供pygame显示
                    tl_detection_results.append({
                        'bbox': (x1, y1, x2, y2),
                        'color': tl_color,
                        'conf': conf,
                        'dist': dist
                    })
                    
                    # 只处理近距离的红绿灯
                    if dist < SAFE_DIST_TRAFFIC_LIGHT:
                        # 根据红绿灯颜色调整目标速度
                        if tl_color == "RED":
                            detected_status = "Red Light - STOP"
                            current_target_v = 0.0  # 红灯停车
                            print(f"[{step}] RED LIGHT detected at {dist:.1f}m - STOP")
                        elif tl_color == "YELLOW":
                            detected_status = "Yellow Light - SLOW DOWN"
                            current_target_v = target_v * 0.5  # 黄灯减速
                            print(f"[{step}] YELLOW LIGHT detected at {dist:.1f}m - SLOW DOWN")
                        elif tl_color == "GREEN":
                            detected_status = "Green Light - GO"
                            current_target_v = target_v  # 绿灯正常行驶
                            print(f"[{step}] GREEN LIGHT detected at {dist:.1f}m - GO")
            
            # --- C. 更新 MPC 目标速度 ---
            dynamic_model.set_target_velocity(current_target_v)
            
            # --- D. MPC 步进 ---
            a_opt, delta_opt, next_state = agent.run_step()

            x, y, yaw, vx, vy, omega = next_state[0]
            trajectory.append([x, y]) 
            velocities.append(vx)  
            accelerations.append(a_opt)  
            steerings.append(delta_opt)  
            times.append(step * simu_step)  
            env.step([a_opt, delta_opt])
            
            if env.display_method == "pygame":
                # update HUD
                env.hud.tick(env, env.clock)
                
                if step == 0:  
                    env.display.fill((0, 0, 0))  
                
                # 绘制HUD基本信息
                env.hud.render(env.display)  
                
                # 绘制红绿灯检测结果
                for tl in tl_detection_results:
                    x1, y1, x2, y2 = tl['bbox']
                    tl_color = tl['color']
                    conf = tl['conf']
                    dist = tl['dist']
                    
                    # 根据红绿灯颜色设置绘制颜色
                    if tl_color == "RED":
                        draw_color = (255, 0, 0)  # 红色
                    elif tl_color == "YELLOW":
                        draw_color = (255, 255, 0)  # 黄色
                    elif tl_color == "GREEN":
                        draw_color = (0, 255, 0)  # 绿色
                    else:
                        draw_color = (128, 128, 128)  # 灰色（未知）
                    
                    # 绘制检测框
                    pygame.draw.rect(env.display, draw_color, pygame.Rect(x1, y1, x2 - x1, y2 - y1), 2)
                    
                    # 绘制标签
                    font = pygame.font.Font(pygame.font.get_default_font(), 14)
                    label = f"TL: {tl_color} ({conf:.2f}) - {dist:.1f}m"
                    text_surface = font.render(label, True, draw_color)
                    text_rect = text_surface.get_rect(bottomleft=(x1, y1 - 5))
                    env.display.blit(text_surface, text_rect)
                
                pygame.display.flip()        
                env.check_quit()

            if np.linalg.norm([next_state[0][0] - agent._end_transform.location.x,
                               next_state[0][1] - agent._end_transform.location.y]) < 1.0:
                print("Destination reached!")
                break 

            if env.display_method == "pygame":
                time.sleep(simu_step)

        except Exception as e:
            print(f"Warning: {e}")
            import traceback
            traceback.print_exc()
            break 

except KeyboardInterrupt:
    print("Simulation interrupted by user.")
finally:
    # 清理传感器资源
    if 'camera_rgb' in locals() and camera_rgb is not None:
        camera_rgb.destroy()
    if 'camera_depth' in locals() and camera_depth is not None:
        camera_depth.destroy()
    print("Sensors cleaned up.")

trajectory = np.array(trajectory)
velocities = np.array(velocities)
accelerations = np.array(accelerations)
steerings = np.array(steerings)
times = np.array(times)

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

axs[0, 0].plot(trajectory[:, 0], trajectory[:, 1], label="Vehicle Path", color='darkorange', linewidth=2)
axs[0, 0].scatter(agent._start_transform.location.x, agent._start_transform.location.y, color='green', label="Start", zorder=5)
axs[0, 0].scatter(agent._end_transform.location.x, agent._end_transform.location.y, color='red', label="End", zorder=5)

route_points = np.array([[wp.transform.location.x, wp.transform.location.y] for wp, _ in route])
axs[0, 0].plot(route_points[:, 0], route_points[:, 1], '--', color='blue', label="Planned Route", alpha=0.6)
axs[0, 0].set_title("Vehicle Path and Planned Route", fontsize=14)
axs[0, 0].set_xlabel("X Position", fontsize=12)
axs[0, 0].set_ylabel("Y Position", fontsize=12)
axs[0, 0].legend(loc='upper left', fontsize=10)
axs[0, 0].grid(True)

axs[0, 1].plot(times, velocities, label="Velocity (m/s)", color='royalblue', linewidth=2)
axs[0, 1].set_title("Velocity over Time", fontsize=14)
axs[0, 1].set_xlabel("Time (s)", fontsize=12)
axs[0, 1].set_ylabel("Velocity (m/s)", fontsize=12)
axs[0, 1].legend(loc='upper right', fontsize=10)
axs[0, 1].grid(True)

axs[1, 0].plot(times, accelerations, label="Acceleration (m/s²)", color='orange', linewidth=2)
axs[1, 0].set_title("Acceleration over Time", fontsize=14)
axs[1, 0].set_xlabel("Time (s)", fontsize=12)
axs[1, 0].set_ylabel("Acceleration (m/s²)", fontsize=12)
axs[1, 0].legend(loc='upper right', fontsize=10)
axs[1, 0].grid(True)

axs[1, 1].plot(times, steerings, label="Steering Angle (rad)", color='green', linewidth=2)
axs[1, 1].set_title("Steering Angle over Time", fontsize=14)
axs[1, 1].set_xlabel("Time (s)", fontsize=12)
axs[1, 1].set_ylabel("Steering Angle (rad)", fontsize=12)
axs[1, 1].legend(loc='upper right', fontsize=10)
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()