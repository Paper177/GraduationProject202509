#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
跟踪器模块
实现卡尔曼滤波器来跟踪车辆距离和速度
"""

import numpy as np

# 导入卡尔曼滤波器库
try:
    from filterpy.kalman import KalmanFilter
    from filterpy.common import Q_discrete_white_noise
except ImportError:
    raise RuntimeError('需要 filterpy, 请运行 "pip install filterpy"')

from .config import DEFAULT_DT, KF_PROCESS_NOISE_VAR, KF_MEASUREMENT_NOISE

class VehicleTracker(object):
    """
    为每个track_id管理一个独立的卡尔曼滤波器，
    用于平滑深度相机的距离测量并估算速度
    
    状态向量: [距离, 速度]
    测量向量: [距离]
    """
    
    def __init__(self, dt=DEFAULT_DT):
        """
        初始化车辆跟踪器
        
        参数:
            dt: 时间步长 (秒)
        """
        self.dt = dt  # 时间步长
        self.filters = {}  # 存储 {track_id: KalmanFilter}
    
    def _create_filter(self, initial_z):
        """
        为新目标创建一个新的卡尔曼滤波器
        
        参数:
            initial_z: 初始距离测量值
            
        返回:
            配置好的卡尔曼滤波器实例
        """
        # 创建一个2状态变量、1测量变量的卡尔曼滤波器
        kf = KalmanFilter(dim_x=2, dim_z=1)
        
        # 1. 初始状态: [测量的距离, 0速度]
        kf.x = np.array([initial_z, 0.0])
        
        # 2. 状态转移矩阵 F (过程模型)
        # [z_k+1]   [1 dt] [z_k]   [0.5*dt^2]
        # [v_k+1] = [0 1 ] [v_k] + [dt      ] * u_k
        # 这里假设加速度u_k为零均值白噪声
        kf.F = np.array([
            [1., self.dt],
            [0., 1.]
        ])
        
        # 3. 测量函数 H (我们只能测量距离)
        kf.H = np.array([[1., 0.]])
        
        # 4. 初始状态协方差 P
        # 表示对初始状态的不确定性
        kf.P *= 10.0  # 初始不确定性较高
        
        # 5. 测量噪声协方差 R
        # 深度相机非常准确，所以我们非常相信测量
        kf.R = np.array([[KF_MEASUREMENT_NOISE]])
        
        # 6. 过程噪声协方差 Q
        # 表示模型不确定性
        kf.Q = Q_discrete_white_noise(dim=2, dt=self.dt, var=KF_PROCESS_NOISE_VAR)
        
        return kf
    
    def get_filter(self, track_id, computed_Z):
        """
        获取或创建指定track_id的卡尔曼滤波器
        
        参数:
            track_id: 目标的跟踪ID
            computed_Z: 当前的距离测量值
            
        返回:
            该track_id对应的卡尔曼滤波器
        """
        if track_id not in self.filters:
            # 如果是新目标，创建一个新的滤波器
            self.filters[track_id] = self._create_filter(computed_Z)
        return self.filters[track_id]
    
    def update_filter(self, track_id, measurement):
        """
        更新指定track_id的滤波器并返回估计状态
        
        参数:
            track_id: 目标的跟踪ID
            measurement: 新的距离测量值
            
        返回:
            (距离估计, 速度估计)
        """
        # 获取滤波器
        kf = self.get_filter(track_id, measurement)
        
        # 预测下一状态
        kf.predict()
        
        # 使用新测量值更新
        kf.update(np.array([measurement]))
        
        # 返回估计的距离和速度
        return kf.x[0], kf.x[1]
    
    def remove_filter(self, track_id):
        """
        移除指定track_id的滤波器，用于清理不再需要的跟踪器
        
        参数:
            track_id: 要移除的跟踪ID
        """
        if track_id in self.filters:
            del self.filters[track_id]
    
    def get_tracked_objects(self):
        """
        获取当前所有被跟踪的目标ID
        
        返回:
            跟踪ID列表
        """
        return list(self.filters.keys())
