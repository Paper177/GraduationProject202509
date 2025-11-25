#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
车辆控制模块
包含 VehicleController 类，增加了安全感知（避障、红绿灯）
"""

import numpy as np
import math
import carla
from collections import deque

class VehicleController(object):
    """
    智能车辆控制器
    集成：纵向PID + 横向Stanley + 安全感知系统 (AEB & 红绿灯)
    """
    
    def __init__(self, args_longitudinal=None, args_lateral=None):
        self.args_longitudinal = args_longitudinal or {'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.05, 'dt': 0.05}
        self.args_lateral = args_lateral or {'k': 0.5, 'k_soft': 1.0}
        self._e_buffer = deque(maxlen=30)
        
        # 安全参数
        self.detection_range = 15.0  # 检测前方多少米
        self.safety_width = 2.5      # 检测宽度（车道宽）

    def run_step(self, vehicle, world, waypoint, target_speed_kmh):
        """
        带安全检查的控制步
        """
        # --- 1. 安全感知层 (Perception & Safety) ---
        hazard_detected, reason = self._detect_hazard(vehicle, world)
        
        if hazard_detected:
            # 如果发现危险（前车、行人、红灯），强制刹车
            return carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0), reason
            
        # --- 2. 控制执行层 (Control) ---
        t = vehicle.get_transform()
        v = vehicle.get_velocity()
        current_speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
        
        throttle, brake = self._pid_longitudinal_control(current_speed, target_speed_kmh)
        steer = self._stanley_lateral_control(t, v, waypoint)
        
        control = carla.VehicleControl()
        control.throttle = throttle
        control.brake = brake
        control.steer = steer
        
        return control, "Normal"

    def _detect_hazard(self, vehicle, world):
        """
        检测红绿灯和前方障碍物
        """
        # A. 红绿灯检测
        if vehicle.is_at_traffic_light():
            traffic_light = vehicle.get_traffic_light()
            if traffic_light and traffic_light.get_state() == carla.TrafficLightState.Red:
                return True, "Red Light"

        # B. 障碍物检测 (简单的空间几何)
        vehicle_transform = vehicle.get_transform()
        vehicle_forward = vehicle_transform.get_forward_vector()
        vehicle_loc = vehicle_transform.location
        
        # 获取周围所有可能碰撞的物体 (Vehicle 和 Walker)
        # 注意：为了性能，这里最好不要每一帧都获取所有actor，但在简单脚本中可以接受
        actors = world.get_actors()
        vehicles = actors.filter('vehicle.*')
        walkers = actors.filter('walker.*')
        obstacles = list(vehicles) + list(walkers)

        for obstacle in obstacles:
            if obstacle.id == vehicle.id:
                continue
            
            # 1. 距离过滤
            obs_loc = obstacle.get_location()
            dist = obs_loc.distance(vehicle_loc)
            
            if dist > self.detection_range:
                continue
                
            # 2. 角度/位置过滤 (是否在车前方)
            # 计算指向障碍物的向量
            vec_to_obs = obs_loc - vehicle_loc
            
            # 计算在车辆前进方向上的投影长度 (纵向距离)
            forward_dist = vec_to_obs.dot(vehicle_forward)
            
            if forward_dist < 0: # 物体在车后方
                continue
                
            # 计算垂直距离 (横向偏差)
            # 使用勾股定理或者向量叉乘
            # 这里简化处理：如果在前方且距离近，再细算
            
            # 简单判断：如果它在前方扇区内
            # 规范化向量
            norm_vec_to_obs = math.sqrt(vec_to_obs.x**2 + vec_to_obs.y**2 + vec_to_obs.z**2)
            if norm_vec_to_obs > 0:
                vec_to_obs_norm = vec_to_obs / norm_vec_to_obs
                dot_prod = vec_to_obs_norm.dot(vehicle_forward)
                
                # 如果夹角很小 (dot_prod 接近 1)，说明在正前方
                # 0.9 约等于 25度角
                if dot_prod > 0.85 and forward_dist < self.detection_range:
                    return True, f"Obstacle: {obstacle.type_id}"
                    
        return False, None

    def _pid_longitudinal_control(self, current_speed, target_speed):
        error = target_speed - current_speed
        self._e_buffer.append(error)

        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self.args_longitudinal['dt']
            _ie = sum(self._e_buffer) * self.args_longitudinal['dt']
        else:
            _de = 0.0
            _ie = 0.0

        output = (self.args_longitudinal['K_P'] * error) + (self.args_longitudinal['K_D'] * _de) + (self.args_longitudinal['K_I'] * _ie)
        
        if output > 0.0:
            return min(output, 1.0), 0.0
        else:
            return 0.0, min(abs(output), 1.0)

    def _stanley_lateral_control(self, vehicle_transform, velocity, target_waypoint):
        current_yaw = vehicle_transform.rotation.yaw
        v_x = velocity.x
        v_y = velocity.y
        current_speed = math.sqrt(v_x**2 + v_y**2)
        
        target_loc = target_waypoint.transform.location
        waypoint_yaw = target_waypoint.transform.rotation.yaw
        
        yaw_diff = math.radians(waypoint_yaw - current_yaw)
        if yaw_diff > math.pi: yaw_diff -= 2 * math.pi
        if yaw_diff < -math.pi: yaw_diff += 2 * math.pi
        
        # 横向误差计算
        front_axle_offset = 1.4
        yaw_rad = math.radians(current_yaw)
        fx = vehicle_transform.location.x + front_axle_offset * math.cos(yaw_rad)
        fy = vehicle_transform.location.y + front_axle_offset * math.sin(yaw_rad)
        dx = fx - target_loc.x
        dy = fy - target_loc.y
        perp_yaw = math.radians(waypoint_yaw + 90)
        crosstrack_error = dx * math.cos(perp_yaw) + dy * math.sin(perp_yaw)

        k = self.args_lateral['k']
        k_soft = self.args_lateral['k_soft']
        steer_correction = math.atan2(k * crosstrack_error, (current_speed + k_soft))
        
        return np.clip((yaw_diff + steer_correction) / 0.61, -1.0, 1.0)