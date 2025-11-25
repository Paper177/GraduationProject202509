#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CARLA YOLO Simulation Package

这个包包含了CARLA模拟器与YOLO目标检测集成的所有组件。
"""

# 包版本
__version__ = "1.0.0"

# 导出主要组件
from .config import YOLO_TARGET_CLASSES, DEFAULT_YOLO_MODEL_PATH, TRACKER_CONFIG
from .tracker import VehicleTracker
from .utils import build_projection_matrix, decode_instance_segmentation, process_depth_image, get_image_point
from .visualization import draw_carla_image, draw_yolo_and_truth, draw_fps_info
from .runner import CarlaYoloRunner
from .main import main, parse_arguments

__all__ = [
    "YOLO_TARGET_CLASSES",
    "DEFAULT_YOLO_MODEL_PATH",
    "TRACKER_CONFIG",
    "VehicleTracker",
    "build_projection_matrix",
    "decode_instance_segmentation",
    "process_depth_image",
    "get_image_point",
    "draw_carla_image",
    "draw_yolo_and_truth",
    "draw_fps_info",
    "CarlaYoloRunner",
    "main",
    "parse_arguments"
]
