#!/usr/bin/env python

import os
import sys

try:
    sys.path.append(os.path.join(
                    os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))), 'official'))
    sys.path.append(os.path.join(
                    os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))), 'utils'))
except IndexError:
    pass

import copy
import carla
import math
import numpy as np
import interpolate as itp
import carla_utils as ca_u
from enum import Enum
from collections import deque
from basic_agent import BasicAgent
from global_route_planner import GlobalRoutePlanner

import matplotlib.pyplot as plt
import time


class RoadOption(Enum):
    """
    道路选项枚举类，表示车辆在道路上的可能行驶状态
    """
    VOID = -1  # 无效选项
    LEFT = 1  # 左转
    RIGHT = 2  # 右转
    STRAIGHT = 3  # 直行
    LANEFOLLOW = 4  # 车道跟随
    CHANGELANELEFT = 5  # 向左变道
    CHANGELANERIGHT = 6  # 向右变道


class Xagent(BasicAgent):
    """
    扩展智能体类，继承自BasicAgent，实现基于MPC的车辆控制
    """
    
    def __init__(self, env, model, dt=0.1) -> None:
        """
        初始化智能体
        
        Args:
            env: 仿真环境对象
            model: 车辆动力学模型
            dt: 仿真时间步长（秒）
        """
        self._env = env  # 仿真环境
        self._vehicle = env.ego_vehicle  # 自车对象
        self._model = model  # 车辆动力学模型
        
        self._world = self._vehicle.get_world()  # CARLA世界对象
        self._map = self._world.get_map()  # CARLA地图对象

        self._base_min_distance = 2.0  # 基础最小距离
        self._waypoints_queue = deque(maxlen=100000)  # 路点队列
        self._d_dist = 0.4  # 路径点间距
        self._sample_resolution = 2.0  # 采样分辨率
        # 变道状态
        self._a_opt = np.array([0.0]*self._model.horizon)  # 最优加速度序列
        self._delta_opt = np.array([0.0]*self._model.horizon)  # 最优转向角序列
        self._dt = dt  # 时间步长

        self._next_states = None  # 下一个状态预测
        self._last_traffic_light = None  # 上一个交通灯
        self._last_traffic_waypoint = None  # 上一个交通灯路点

        # 设置MPC求解器权重矩阵
        self._model.solver_basis(Q=np.diag([10, 10, 10, 1.5, 0.1]), Rd=np.diag([1.0, 1000.0]))
        self.Q_origin = copy.deepcopy(self._model.Q)  # 原始Q矩阵备份
        self._log_data = []  # 日志数据
        self._simu_time = 0  # 仿真时间
        
        # 初始化全局路径规划器
        self._global_planner = GlobalRoutePlanner(self._map, self._sample_resolution)
        
        self.dist_move = 0.2  # 初始移动距离
        self.dist_step = 1.5  # 距离步长系数

    def plan_route(self, start_location, end_location):
        """
        规划从起点到终点的路径，并将路点加入队列
        
        Args:
            start_location: 起点位置
            end_location: 终点位置
        """
        self._route = self.trace_route(start_location.location, end_location.location)
        for i in self._route:
            self._waypoints_queue.append(i)
    
    def set_start_end_transforms(self, start_idx, end_idx):
        """
        设置起点和终点的变换
        
        Args:
            start_idx: 起点索引
            end_idx: 终点索引
        
        Raises:
            IndexError: 索引超出范围
        """
        spawn_points = self._map.get_spawn_points()  # 获取所有生成点
        if start_idx < len(spawn_points) and end_idx < len(spawn_points):
            self._start_transform = spawn_points[start_idx]  # 设置起始点变换
            self._end_transform = spawn_points[end_idx]  # 设置终止点变换
        else:
            raise IndexError("Start or end index out of bounds!")

    def calc_ref_trajectory_in_T_step(self, node, ref_path, sp):
        """
        计算T步长内的参考轨迹
        
        Args:
            node: 当前节点信息 [x, y, v, yaw]
            ref_path: 参考路径对象
            sp: 速度剖面
            
        Returns:
            z_ref: 参考轨迹 [x, y, v, yaw]
            ind: 目标索引
        """
        T = self._model.horizon  # MPC预测时域
        z_ref = np.zeros((4, T + 1))  # 参考轨迹数组
        length = ref_path.length  # 参考路径长度
        ind, _ = ref_path.nearest_index(node)  # 找到最近的路径点索引

        # 初始化参考轨迹起点
        z_ref[0, 0] = ref_path.cx[ind]  # x坐标
        z_ref[1, 0] = ref_path.cy[ind]  # y坐标
        z_ref[2, 0] = sp[ind]  # 速度
        z_ref[3, 0] = ref_path.cyaw[ind]  # 航向角

        dist_move = copy.copy(self.dist_move)  # 复制初始移动距离

        # 计算未来T步的参考轨迹
        for i in range(1, T + 1):
            # 根据当前速度更新移动距离
            dist_move += self.dist_step * abs(self._model.get_v()) * self._dt
            ind_move = int(round(dist_move / self._d_dist))  # 移动的索引数
            index = min(ind + ind_move, length - 1)  # 确保索引不超出范围

            # 更新参考轨迹
            z_ref[0, i] = ref_path.cx[index]
            z_ref[1, i] = ref_path.cy[index]
            z_ref[2, i] = sp[index]
            z_ref[3, i] = ref_path.cyaw[index]

        return z_ref, ind

    def rotate(self, x, y, theta, ratio=1.75):
        """
        旋转坐标点
        
        Args:
            x: x坐标
            y: y坐标
            theta: 旋转角度（弧度）
            ratio: 缩放比例
            
        Returns:
            旋转后的坐标点
        """
        return np.array([(x * np.cos(theta) - y * np.sin(theta)) * ratio, 
                        (x * np.sin(theta) + y * np.cos(theta)) * ratio])
    
    def lat_dis_wp_ev(self, wp, ev):
        """
        计算路点与自车之间的横向距离
        
        Args:
            wp: 路点对象
            ev: 自车对象
            
        Returns:
            横向距离绝对值
        """
        wp_loc = np.array([wp.transform.location.x, wp.transform.location.y])  # 路点位置
        ev_loc = np.array([ev.get_location().x, ev.get_location().y])  # 自车位置
        wp_yaw = wp.transform.rotation.yaw  # 路点航向角
        # 旋转坐标到路点坐标系
        wp_loc = self.rotate(wp_loc[0], wp_loc[1], np.deg2rad(wp_yaw))
        ev_loc = self.rotate(ev_loc[0], ev_loc[1], np.deg2rad(wp_yaw))
        return np.abs(wp_loc[1] - ev_loc[1])  # 返回横向距离绝对值
    
    def run_step(self, lv=None):
        """
        执行一步智能体控制
        
        Args:
            lv: 可选的领先车辆信息
            
        Returns:
            a_opt[0]: 最优加速度
            delta_opt[0]: 最优转向角
            next_state: 下一个状态
        """
        self._simu_time += self._dt  # 更新仿真时间
        # 获取车辆当前状态和高度
        state, height = self._model.get_state_carla() 
        # 转换为右手坐标系
        current_state = np.array(ca_u.carla_vector_to_rh_vector(state[0:2], state[2], state[3:]))
        
        # 清除过时的路点
        veh_location = self._vehicle.get_location()  # 自车位置
        vehicle_speed = self._model.get_v()  # 自车速度
        # 动态调整最小距离
        self._min_distance = self._base_min_distance + 0.5 * vehicle_speed

        # 获取路点
        if len(self._waypoints_queue) == 0:
            raise Exception("No waypoints to follow")
        else:
            carla_wp, _ = np.array(self._waypoints_queue).T  # 路点和道路选项
            waypoints = []  # 存储转换后的路点
            v = math.sqrt(current_state[3]**2 + current_state[4]**2)  # 当前速度
            # 添加当前位置作为第一个路点
            waypoints.append([current_state[0], current_state[1], v, current_state[2]])
            cnt = 0  # 路点计数

            # 删除重复路点，避免样条插值出现NaN
            last_state = None  # 上一个路点状态
            for wp in carla_wp:
                if cnt > 30:  # 最多保存30个路点
                    break
                cnt += 1
                t = wp.transform  # 路点变换
                # 转换为右手坐标系
                ref_state = ca_u.carla_vector_to_rh_vector(
                    [t.location.x, t.location.y], t.rotation.yaw)
                # 检查与上一个路点的距离，避免重复
                if last_state is not None:
                    if np.sqrt(ref_state[0]**2 + ref_state[1]**2) - last_state < 0.005:
                        continue
                # 添加路点到列表
                waypoints.append([ref_state[0], ref_state[1],
                                 self._model.target_v, ref_state[2]])
                last_state = np.sqrt(ref_state[0]**2 + ref_state[1]**2)

            waypoints = np.array(waypoints).T  # 转置路点数组

        num_waypoint_removed = 0  # 要移除的路点数量

        # 移除已经通过的路点
        for waypoint, _ in self._waypoints_queue:
            # 最后一个路点保留到非常接近时才移除
            if len(self._waypoints_queue) - num_waypoint_removed == 1:
                min_distance = 1
            else:
                min_distance = self._min_distance

            # 如果路点距离小于最小距离，标记为移除
            if veh_location.distance(waypoint.transform.location) < min_distance:
                num_waypoint_removed += 1
            else:
                break

        # 从队列中移除标记的路点
        if num_waypoint_removed > 0:
            for _ in range(num_waypoint_removed):
                self._waypoints_queue.popleft()

        # 插值生成平滑路径
        cx, cy, cyaw, ck, s = itp.calc_spline_course_carla(
            waypoints[0], waypoints[1], waypoints[3][0], ds=self._d_dist)
        # 计算速度剖面
        sp = itp.calc_speed_profile(cx, cy, cyaw, self._model.target_v)

        ref_path = itp.PATH(cx, cy, cyaw, ck)  # 创建参考路径对象
        # 计算参考轨迹
        z_ref, target_ind = self.calc_ref_trajectory_in_T_step(
            [current_state[0], current_state[1], v, current_state[2]], ref_path, sp)
        # 构建参考轨迹数组
        ref_traj = np.array([z_ref[0], z_ref[1], z_ref[3], z_ref[2], [
                            0]*len(z_ref[0]), [0]*len(z_ref[0])])[:, :self._model.horizon]

        # 初始化下一个状态数组
        if self._next_states is None:
            self._next_states = np.zeros(
                (self._model.n_states, self._model.horizon+1)).T
        
        cur_v = self._model.get_v()  # 当前速度
        self._next_states[:, 3] = cur_v  # 更新速度
        current_state[3:] = self._next_states[0][3:]  # 更新当前状态
        # 构建控制输入初始值
        u0 = np.array([self._a_opt, self._delta_opt]).reshape(-1, 2).T

        # 再次检查并初始化下一个状态数组
        if self._next_states is None:
            self._next_states = np.zeros(
                (self._model.n_states, self._model.horizon+1)).T
        
        apf_obs = apf_nc_road = apf_c_road = 0  # 人工势场相关变量
        self._model.solver_add_cost()  # 添加成本函数
        self._model.solver_add_bounds()  # 添加约束条件
        
        # 求解MPC
        tick = time.time()
        state = self._model.solve_MPC(
            ref_traj.T, current_state, self._next_states, u0)
        time_2 = time.time() - tick  # 求解时间
        
        # 在CARLA中绘制规划轨迹
        ca_u.draw_planned_trj(self._world, ref_traj[:2, :].T, height+0.5)
        ca_u.draw_planned_trj(self._world, state[2][:, :2], height+0.5, color=(0, 233, 222))

        self._next_states = state[2]  # 更新下一个状态预测

        next_state = state[2][1]  # 下一个状态
        self._a_opt = state[0]  # 更新最优加速度序列
        self._delta_opt = state[1]  # 更新最优转向角序列
        
        # 预测下一个状态
        next_state = self._model.predict(current_state, (self._a_opt[0], self._delta_opt[0]))
        self._model.set_state(next_state)  # 更新模型状态

        # 计算距离误差和航向误差
        dist_error = math.hypot(
            next_state[0] - ref_traj[0, 1], next_state[1] - ref_traj[1, 1] - self.dist_move)
        yaw_error = abs(next_state[2] - ref_traj[2, 1])
        
        # 返回最优控制量和下一个状态
        return self._a_opt[0], self._delta_opt[0], (next_state, height+0.05)

    def trace_route(self, start_location, end_location):
        """
        追踪从起点到终点的路径
        
        Args:
            start_location: 起点位置
            end_location: 终点位置
            
        Returns:
            route: 路径点列表
        """
        return self._global_planner.trace_route(start_location, end_location)