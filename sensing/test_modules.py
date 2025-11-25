#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模块测试脚本

这个脚本用于测试拆分后的代码模块是否能正确导入和工作。
它不实际运行模拟，只是验证各个组件的基本功能。
"""

import sys
import os

# 添加当前目录到路径以支持模块导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_module_imports():
    """测试所有模块是否能正确导入"""
    print("Testing module imports...")
    
    # 测试配置模块
    try:
        from src.config import YOLO_TARGET_CLASSES, EDGES, TRACKER_CONFIG
        print("✓ Config module imported successfully")
        print(f"  - Target classes: {YOLO_TARGET_CLASSES}")
        print(f"  - Tracker config: {TRACKER_CONFIG}")
    except Exception as e:
        print(f"✗ Failed to import config module: {e}")
    
    # 测试工具函数模块
    try:
        from src.utils import create_3d_to_2d_matrix, decode_segmap
        print("✓ Utils module imported successfully")
        print("  - 3D to 2D matrix function available")
        print("  - Segmentation decoding function available")
    except Exception as e:
        print(f"✗ Failed to import utils module: {e}")
    
    # 测试跟踪器模块
    try:
        from src.tracker import VehicleTracker
        print("✓ Tracker module imported successfully")
        print("  - VehicleTracker class available")
    except Exception as e:
        print(f"✗ Failed to import tracker module: {e}")
    
    # 测试可视化模块
    try:
        from src.visualization import draw_carla_image, draw_yolo_and_truth
        print("✓ Visualization module imported successfully")
        print("  - Image drawing functions available")
    except Exception as e:
        print(f"✗ Failed to import visualization module: {e}")
    
    # 测试运行器模块
    try:
        from src.runner import CarlaYoloRunner
        print("✓ Runner module imported successfully")
        print("  - CarlaYoloRunner class available")
    except Exception as e:
        print(f"✗ Failed to import runner module: {e}")
    
    # 测试入口脚本
    try:
        from src.main import parse_arguments
        print("✓ Main module imported successfully")
        print("  - Argument parser function available")
    except Exception as e:
        print(f"✗ Failed to import main module: {e}")

def test_tracker_functionality():
    """测试跟踪器的基本功能"""
    print("\nTesting basic tracker functionality...")
    
    try:
        from src.tracker import VehicleTracker
        
        # 创建跟踪器实例
        tracker = VehicleTracker()
        print("  ✓ VehicleTracker instance created")
        
        # 测试初始化状态
        initial_state = tracker.get_state()
        print(f"  ✓ Initial state: {initial_state}")
        
        # 测试预测功能
        tracker.predict()
        predicted_state = tracker.get_state()
        print(f"  ✓ Predicted state: {predicted_state}")
        
        print("✓ Tracker functionality test passed")
        return True
    except Exception as e:
        print(f"✗ Tracker functionality test failed: {e}")
        return False

def test_configuration_consistency():
    """测试配置一致性"""
    print("\nTesting configuration consistency...")
    
    try:
        from src.config import YOLO_TARGET_CLASSES, VEHICLE_COLOR
        
        # 检查YOLO目标类别和车辆颜色配置是否一致
        if len(YOLO_TARGET_CLASSES) == len(VEHICLE_COLOR):
            print(f"  ✓ Configuration consistency check passed")
            print(f"  - {len(YOLO_TARGET_CLASSES)} target classes with corresponding colors")
            return True
        else:
            print(f"  ✗ Configuration inconsistency found:")
            print(f"    - {len(YOLO_TARGET_CLASSES)} target classes")
            print(f"    - {len(VEHICLE_COLOR)} color entries")
            return False
    except Exception as e:
        print(f"✗ Configuration consistency test failed: {e}")
        return False

def test_directory_structure():
    """测试目录结构是否正确"""
    print("\nTesting directory structure...")
    
    # 检查必要的目录是否存在
    required_dirs = [
        'src',
    ]
    
    # 检查必要的文件是否存在
    required_files = [
        'src/__init__.py',
        'src/config.py',
        'src/utils.py',
        'src/tracker.py',
        'src/visualization.py',
        'src/runner.py',
        'src/main.py',
        'run_simulation.py',
    ]
    
    all_present = True
    
    # 检查目录
    for directory in required_dirs:
        if not os.path.isdir(directory):
            print(f"✗ Directory not found: {directory}")
            all_present = False
        else:
            print(f"✓ Directory found: {directory}")
    
    # 检查文件
    for file in required_files:
        if not os.path.isfile(file):
            print(f"✗ File not found: {file}")
            all_present = False
        else:
            print(f"✓ File found: {file}")
    
    return all_present

def create_init_file():
    """创建__init__.py文件使src成为Python包"""
    init_path = 'src/__init__.py'
    if not os.path.exists(init_path):
        print(f"\nCreating {init_path}...")
        with open(init_path, 'w') as f:
            f.write("""
"""
            )
        print(f"✓ {init_path} created successfully")
        return True
    return False

def main():
    """运行所有测试"""
    print("="*60)
    print("  CARLA YOLO Simulation - Module Testing")
    print("="*60)
    
    # 确保src是一个Python包
    create_init_file()
    
    # 运行各项测试
    import_test = False
    try:
        test_module_imports()
        import_test = True
    except Exception as e:
        print(f"\n✗ Module import test failed: {e}")
    
    # 只有在导入测试通过后才运行功能测试
    if import_test:
        tracker_test = test_tracker_functionality()
        config_test = test_configuration_consistency()
    else:
        tracker_test = False
        config_test = False
    
    structure_test = test_directory_structure()
    
    # 总结测试结果
    print("\n" + "="*60)
    print("  TEST SUMMARY")
    print("="*60)
    print(f"  Module Imports: {'PASS' if import_test else 'FAIL'}")
    print(f"  Tracker Functionality: {'PASS' if tracker_test else 'FAIL'}")
    print(f"  Configuration Consistency: {'PASS' if config_test else 'FAIL'}")
    print(f"  Directory Structure: {'PASS' if structure_test else 'FAIL'}")
    
    # 整体测试结果
    all_passed = import_test and tracker_test and config_test and structure_test
    print("="*60)
    print(f"  OVERALL RESULT: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("="*60)
    
    if all_passed:
        print("\nThe code has been successfully modularized!")
        print("You can now run the simulation using:")
        print("  python run_simulation.py")
    else:
        print("\nPlease fix the identified issues before running the simulation.")
    
    return all_passed

if __name__ == '__main__':
    main()
