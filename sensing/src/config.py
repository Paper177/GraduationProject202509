#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置和常量模块
定义所有全局常量、配置参数和默认值
"""

# YOLO目标检测相关配置
YOLO_TARGET_CLASSES = ['car', 'truck', 'bus', 'motorcycle', 'person', 'bicycle']
DEFAULT_YOLO_MODEL_PATH = 'yolo11s.pt'  # 默认YOLO模型路径
DEFAULT_WORLD_NAME = 'Town05'  # 默认世界名称



# 可视化配置
VEHICLE_COLOR = (0, 0, 255)  # 蓝色，用于绘制车辆边界框
PEDESTRIAN_COLOR = (255, 0, 0)  # 红色，用于绘制行人边界框
BICYCLE_COLOR = (0, 255, 0)  # 绿色，用于绘制自行车边界框
EDGES = [  # 3D边界框的边缘定义
    [0, 1], [1, 3], [3, 2], [2, 0],  # 底面
    [0, 4], [4, 5], [5, 1],          # 前面
    [5, 7], [7, 6], [6, 4],          # 顶面
    [6, 2], [7, 3]                   # 后面
]

# 卡尔曼滤波器配置
DEFAULT_DT = 0.05  # 默认时间步长 (20 FPS)
KF_PROCESS_NOISE_VAR = 0.1  # 过程噪声方差
KF_MEASUREMENT_NOISE = 0.01  # 测量噪声

# 卡尔曼滤波器跟踪器配置
TRACKER_CONFIG = {
    'process_noise': KF_PROCESS_NOISE_VAR,
    'measurement_noise': KF_MEASUREMENT_NOISE,
    'dt': DEFAULT_DT
}

# 相机配置默认值
DEFAULT_FOV = 70.0  # 默认视场角
DEFAULT_CAMERA_LOCATION = (1.2, 0.0, 2.0)  # 默认相机位置 (x, y, z)

# 性能配置
TARGET_FPS = 20  # 目标帧率
DEFAULT_TIMEOUT = 1.0  # 默认队列超时时间

# 场景配置
DEFAULT_MAP = DEFAULT_WORLD_NAME  # 默认地图
DEFAULT_NUM_VEHICLES = 50  # 默认NPC车辆数量
DEFAULT_NUM_WALKERS =50  # 默认NPC行人数量

# 窗口配置
DEFAULT_RESOLUTION = '1280x720'  # 默认窗口分辨率

# 日志配置
LOG_FORMAT = '%(levelname)s: %(message)s'
LOG_LEVEL = 'INFO'
