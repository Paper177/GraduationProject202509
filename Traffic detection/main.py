import argparse
import logging
from simulation import CarlaYoloRunner

def main():
    argparser = argparse.ArgumentParser(description='CARLA (YOLO+深度) 3D 速度检测')
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='主机IP')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP端口')
    argparser.add_argument('--res', metavar='WIDTHxHEIGHT', default='1280x720', help='窗口分辨率')
    argparser.add_argument('--yolo-model', metavar='PATH', required=True, help='YOLO 模型路径')
    argparser.add_argument('--tm-port', metavar='P', default=8000, type=int, help='交通管理器端口')
    argparser.add_argument('-n', '--num-vehicles', metavar='N', default=30, type=int, help='NPC车辆数量')
    argparser.add_argument('-w', '--num-walkers', metavar='W', default=15, type=int, help='NPC行人数量')
    
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]
    
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    runner = None
    try:
        runner = CarlaYoloRunner(args)
        runner.run()
    except Exception as e:
        logging.critical(f"程序启动失败: {e}")

if __name__ == '__main__':
    main()