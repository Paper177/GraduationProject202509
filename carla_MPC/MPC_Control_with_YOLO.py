import numpy as np
import matplotlib.pyplot as plt
from src.mcp_controller import Vehicle
from env import Env, draw_waypoints

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).with_name('src')))
from src.x_v2x_agent import Xagent
from src.global_route_planner import GlobalRoutePlanner

import time
import pygame


class MPCCarSimulation:
    """MPC车辆控制仿真主类"""
    
    def __init__(self):
        """初始化仿真参数和环境"""
        # 仿真参数配置
        self.simulation_params = {
            'time_step': 0.05,  # 仿真时间步长（秒）
            'target_speed': 60,  # 目标速度（km/h）
            'sample_resolution': 2.0,  # 路径规划采样分辨率
            'display_mode': "pygame",  # 显示模式："spec" 或 "pygame"
            'max_simulation_steps': 5000,  # 最大仿真步数
            'destination_threshold': 1.0  # 到达目的地的距离阈值（米）
        }
        
        # 初始化环境
        self.env = self._initialize_environment()
        self.spawn_points = self.env.map.get_spawn_points()
        
        # 仿真数据记录
        self.simulation_data = {
            'trajectory': [],
            'velocities': [],
            'accelerations': [],
            'steerings': [],
            'times': []
        }
        
        # 车辆和控制器
        self.vehicle = None
        self.agent = None
        self.route = None
        
    def _initialize_environment(self):
        """初始化仿真环境"""
        env = Env(
            display_method=self.simulation_params['display_mode'],
            dt=self.simulation_params['time_step']
        )
        env.clean()
        return env
    
    def _setup_route(self, start_idx, end_idx):
        """设置起点和终点，规划全局路径"""
        # 创建全局路径规划器
        route_planner = GlobalRoutePlanner(
            self.env.map, 
            self.simulation_params['sample_resolution']
        )
        
        # 规划路径
        self.route = route_planner.trace_route(
            self.spawn_points[start_idx].location, 
            self.spawn_points[end_idx].location
        )
        
        # 可视化路径点
        draw_waypoints(
            self.env.world, 
            [wp for wp, _ in self.route], 
            z=0.5, 
            color=(0, 255, 0)
        )
        
        # 重置环境到起点
        self.env.reset(spawn_point=self.spawn_points[start_idx])
        
    def _initialize_vehicle_and_agent(self, start_idx, end_idx):
        """初始化车辆动力学模型和智能体"""
        # 初始化车辆动力学模型
        self.vehicle = Vehicle(
            actor=self.env.ego_vehicle,
            horizon=10,
            target_v=self.simulation_params['target_speed'],
            delta_t=self.simulation_params['time_step'],
            max_iter=30
        )
        
        # 初始化智能体
        self.agent = Xagent(
            self.env, 
            self.vehicle, 
            dt=self.simulation_params['time_step']
        )
        self.agent.set_start_end_transforms(start_idx, end_idx)
        self.agent.plan_route(self.agent._start_transform, self.agent._end_transform)
    
    def _update_pygame_display(self, step):
        """更新Pygame显示"""
        # 更新HUD
        self.env.hud.tick(self.env, self.env.clock)
        
        # 初始化显示
        if step == 0:
            self.env.display.fill((0, 0, 0))
        
        # 渲染HUD并刷新显示
        self.env.hud.render(self.env.display)
        pygame.display.flip()
        
        # 检查退出事件
        self.env.check_quit()
        
        # 控制仿真速度
        time.sleep(self.simulation_params['time_step'])
    
    def _check_destination_reached(self, next_state):
        """检查是否到达目的地"""
        x, y = next_state[0][0], next_state[0][1]
        end_x, end_y = self.agent._end_transform.location.x, self.agent._end_transform.location.y
        
        distance = np.linalg.norm([x - end_x, y - end_y])
        return distance < self.simulation_params['destination_threshold']
    
    def _record_simulation_data(self, step, next_state, acceleration, steering):
        """记录仿真数据"""
        x, y, _, vx, _, _ = next_state[0]
        current_time = step * self.simulation_params['time_step']
        
        self.simulation_data['trajectory'].append([x, y])
        self.simulation_data['velocities'].append(vx)
        self.simulation_data['accelerations'].append(acceleration)
        self.simulation_data['steerings'].append(steering)
        self.simulation_data['times'].append(current_time)
    
    def _visualize_results(self):
        """可视化仿真结果"""
        # 将列表转换为numpy数组
        trajectory = np.array(self.simulation_data['trajectory'])
        velocities = np.array(self.simulation_data['velocities'])
        accelerations = np.array(self.simulation_data['accelerations'])
        steerings = np.array(self.simulation_data['steerings'])
        times = np.array(self.simulation_data['times'])
        
        # 创建2x2子图
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        
        # 1. 车辆轨迹和规划路径
        axs[0, 0].plot(trajectory[:, 0], trajectory[:, 1], 
                      label="Vehicle Path", color='darkorange', linewidth=2)
        axs[0, 0].scatter(self.agent._start_transform.location.x, 
                         self.agent._start_transform.location.y, 
                         color='green', label="Start", zorder=5)
        axs[0, 0].scatter(self.agent._end_transform.location.x, 
                         self.agent._end_transform.location.y, 
                         color='red', label="End", zorder=5)
        
        # 绘制规划路径
        route_points = np.array([[wp.transform.location.x, wp.transform.location.y] 
                                for wp, _ in self.route])
        axs[0, 0].plot(route_points[:, 0], route_points[:, 1], 
                      '--', color='blue', label="Planned Route", alpha=0.6)
        
        axs[0, 0].set_title("Vehicle Path and Planned Route", fontsize=14)
        axs[0, 0].set_xlabel("X Position", fontsize=12)
        axs[0, 0].set_ylabel("Y Position", fontsize=12)
        axs[0, 0].legend(loc='upper left', fontsize=10)
        axs[0, 0].grid(True)
        
        # 2. 速度-时间曲线
        axs[0, 1].plot(times, velocities, 
                      label="Velocity (m/s)", color='royalblue', linewidth=2)
        axs[0, 1].set_title("Velocity over Time", fontsize=14)
        axs[0, 1].set_xlabel("Time (s)", fontsize=12)
        axs[0, 1].set_ylabel("Velocity (m/s)", fontsize=12)
        axs[0, 1].legend(loc='upper right', fontsize=10)
        axs[0, 1].grid(True)
        
        # 3. 加速度-时间曲线
        axs[1, 0].plot(times, accelerations, 
                      label="Acceleration (m/s²)", color='orange', linewidth=2)
        axs[1, 0].set_title("Acceleration over Time", fontsize=14)
        axs[1, 0].set_xlabel("Time (s)", fontsize=12)
        axs[1, 0].set_ylabel("Acceleration (m/s²)", fontsize=12)
        axs[1, 0].legend(loc='upper right', fontsize=10)
        axs[1, 0].grid(True)
        
        # 4. 转向角-时间曲线
        axs[1, 1].plot(times, steerings, 
                      label="Steering Angle (rad)", color='green', linewidth=2)
        axs[1, 1].set_title("Steering Angle over Time", fontsize=14)
        axs[1, 1].set_xlabel("Time (s)", fontsize=12)
        axs[1, 1].set_ylabel("Steering Angle (rad)", fontsize=12)
        axs[1, 1].legend(loc='upper right', fontsize=10)
        axs[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def run_simulation(self, start_idx=87, end_idx=40):
        """运行完整的仿真流程
        
        Args:
            start_idx: 起点索引
            end_idx: 终点索引
        """
        try:
            # 设置路径
            self._setup_route(start_idx, end_idx)
            
            # 初始化车辆和智能体
            self._initialize_vehicle_and_agent(start_idx, end_idx)
            
            # 初始化Pygame显示（如果需要）
            if self.env.display_method == "pygame":
                self.env.init_display()
            
            # 主仿真循环
            for step in range(self.simulation_params['max_simulation_steps']):
                try:
                    # 执行一步MPC控制
                    acceleration_opt, steering_opt, next_state = self.agent.run_step()
                    
                    # 记录仿真数据
                    self._record_simulation_data(
                        step, next_state, acceleration_opt, steering_opt
                    )
                    
                    # 执行环境步进
                    self.env.step([acceleration_opt, steering_opt])
                    
                    # 更新Pygame显示
                    if self.env.display_method == "pygame":
                        self._update_pygame_display(step)
                    
                    # 检查是否到达目的地
                    if self._check_destination_reached(next_state):
                        print("Destination reached!")
                        if self.env.display_method == "pygame":
                            pygame.quit()
                        break
                    
                except Exception as e:
                    print(f"Simulation step error: {e}")
                    continue
            
        except KeyboardInterrupt:
            print("Simulation interrupted by user.")
        except Exception as e:
            print(f"Critical simulation error: {e}")
        finally:
            # 可视化结果
            self._visualize_results()


def main():
    """主函数"""
    # 创建仿真实例
    simulation = MPCCarSimulation()
    
    # 运行仿真
    simulation.run_simulation()


if __name__ == "__main__":
    main()