#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CARLA YOLO 模拟主运行脚本

这个脚本是原始 run_simulation_with_yolo.py 的优化版本，
它通过引用 src 目录中的模块化组件来实现相同的功能，
使代码更易于维护和调试。
"""

import sys
import os

# 确保可以导入src目录中的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入模块化组件
from src.main import main

if __name__ == '__main__':
    # 调用主函数启动模拟
    main()
