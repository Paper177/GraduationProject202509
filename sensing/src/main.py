#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入必要的模块
from src.runner import CarlaYoloRunner
from src.config import DEFAULT_YOLO_MODEL_PATH, DEFAULT_WORLD_NAME, DEFAULT_NUM_VEHICLES, DEFAULT_NUM_WALKERS

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CARLA Simulation with YOLO Integration')
    
    # YOLO模型参数
    parser.add_argument('--yolo-model', type=str, default=DEFAULT_YOLO_MODEL_PATH,
                        help='Path to YOLO model weights')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='Confidence threshold for YOLO detection')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                        help='IOU threshold for YOLO detection')
    
    # CARLA环境参数
    parser.add_argument('--host', type=str, default='localhost',
                        help='CARLA server hostname')
    parser.add_argument('--port', type=int, default=2000,
                        help='CARLA server port')
    parser.add_argument('--world-name', type=str, default=DEFAULT_WORLD_NAME,
                        help='Name of the CARLA world to load')
    parser.add_argument('--tm-port', type=int, default=8000,
                        help='Traffic manager port')
    parser.add_argument('--no-rendering', action='store_true',
                        help='Disable rendering to improve performance')
    
    # 模拟参数
    parser.add_argument('--duration', type=int, default=300,
                        help='Duration of the simulation in seconds')
    parser.add_argument('--num-vehicles', type=int, default=DEFAULT_NUM_VEHICLES,
                        help='Number of NPC vehicles to spawn')
    parser.add_argument('--num-walkers', type=int, default=DEFAULT_NUM_WALKERS,
                        help='Number of NPC pedestrians to spawn')
    parser.add_argument('--record', action='store_true',
                        help='Record simulation data')
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='Directory to save output data')
    
    return parser.parse_args()

def main():
    """主函数"""
    try:
        # 解析命令行参数
        args = parse_arguments()
        
        # 确保输出目录存在
        if args.record and not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            logger.info(f"Created output directory: {args.output_dir}")
        
        # 创建并初始化CarlaYoloRunner
        runner = CarlaYoloRunner(
            yolo_model_path=args.yolo_model,
            conf_threshold=args.conf_thres,
            iou_threshold=args.iou_thres,
            carla_host=args.host,
            carla_port=args.port,
            world_name=args.world_name,
            tm_port=args.tm_port,
            disable_rendering=args.no_rendering,
            record_data=args.record,
            output_directory=args.output_dir,
            num_vehicles=args.num_vehicles,
            num_walkers=args.num_walkers
        )
        
        # 运行模拟
        logger.info("Starting CARLA simulation with YOLO integration...")
        runner.run(duration=args.duration)
        
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise
    finally:
        # 确保清理资源
        if 'runner' in locals():
            logger.info("Cleaning up resources...")
            runner.cleanup()
        logger.info("Simulation finished")

if __name__ == '__main__':
    main()
