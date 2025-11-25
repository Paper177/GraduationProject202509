#!/usr/bin/env python
# -*- coding: utf-8 -*-

import carla
import queue
import pygame
import logging
import random
import numpy as np
import time
import math

try:
    from ultralytics import YOLO
except ImportError:
    raise RuntimeError('cannot import ultralytics')

from .config import DEFAULT_DT, TARGET_FPS, DEFAULT_CAMERA_LOCATION
from .utils import build_projection_matrix, process_depth_image, decode_instance_segmentation
from .tracker import VehicleTracker
from .visualization import draw_carla_image, draw_yolo_and_truth, draw_fps_info
from .controller import VehicleController
from pygame.locals import K_ESCAPE

class CarlaYoloRunner(object):
    def __init__(self, yolo_model_path, conf_threshold=0.25, iou_threshold=0.45,
                 carla_host='localhost', carla_port=2000, world_name='Town03',
                 disable_rendering=False, record_data=False, output_directory='./output',
                 width=800, height=600, tm_port=8000, num_vehicles=50, num_walkers=30):
        
        self.width = width
        self.height = height
        self.run_simulation = True
        self.yolo_model_path = yolo_model_path
        self.carla_host = carla_host
        self.carla_port = carla_port
        self.world_name = world_name
        self.tm_port = tm_port
        self.num_vehicles = num_vehicles
        self.num_walkers = num_walkers
        
        self.display = None
        self.clock = None
        self.yolo_font = None
        
        self.client = None
        self.world = None
        self.map = None
        self.ego_vehicle = None
        
        self.camera_rgb = None
        self.camera_depth = None
        self.inst_camera = None
        
        self.rgb_queue = queue.Queue()
        self.depth_queue = queue.Queue()
        self.inst_queue = queue.Queue()
        
        self.traffic_manager = None
        self.actors_to_destroy = []
        self.walker_controllers = []
        
        self.model = None
        self.last_inference_time = 0.0
        self.vehicle_tracker = VehicleTracker(dt=DEFAULT_DT)
        self.controller = VehicleController()
        
        self.jaywalking_timer = 0

    def _init_pygame(self):
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode((self.width, self.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()
        self.yolo_font = pygame.font.SysFont("Arial", 16)

    def _load_yolo_model(self):
        logging.info(f"Loading YOLO: {self.yolo_model_path}")
        self.model = YOLO(self.yolo_model_path)

    def _setup_carla(self):
        self.client = carla.Client(self.carla_host, self.carla_port)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(self.world_name)
        self.map = self.world.get_map()
        self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.global_percentage_speed_difference(-10.0) 
        
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / TARGET_FPS
        self.world.apply_settings(settings)
        self.world.tick()

    def _spawn_actors(self):
        logging.info("Spawning Actors...")
        bp_lib = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()

        # 1. 主角
        vehicle_bp = bp_lib.find('vehicle.tesla.cybertruck')
        vehicle_bp.set_attribute('role_name', 'ego')
        if spawn_points:
            transform = spawn_points.pop(0) # 拿走第一个点
        else:
            transform = carla.Transform(carla.Location(x=0,y=0,z=2))
            
        self.ego_vehicle = self.world.try_spawn_actor(vehicle_bp, transform)
        self.ego_vehicle.set_autopilot(False) # 确保关闭内置AI
        self.actors_to_destroy.append(self.ego_vehicle)

        # 2. 传感器
        fov = 70.0
        cam_trans = carla.Transform(carla.Location(x=DEFAULT_CAMERA_LOCATION[0], y=DEFAULT_CAMERA_LOCATION[1], z=DEFAULT_CAMERA_LOCATION[2]))
        
        self.camera_bp_rgb = bp_lib.find('sensor.camera.rgb')
        self.camera_bp_rgb.set_attribute('image_size_x', str(self.width))
        self.camera_bp_rgb.set_attribute('image_size_y', str(self.height))
        self.camera_bp_rgb.set_attribute('fov', str(fov))
        self.camera_rgb = self.world.spawn_actor(self.camera_bp_rgb, cam_trans, attach_to=self.ego_vehicle)
        self.camera_rgb.listen(self.rgb_queue.put)
        self.actors_to_destroy.append(self.camera_rgb)

        cam_bp_depth = bp_lib.find('sensor.camera.depth')
        cam_bp_depth.set_attribute('image_size_x', str(self.width))
        cam_bp_depth.set_attribute('image_size_y', str(self.height))
        cam_bp_depth.set_attribute('fov', str(fov))
        self.camera_depth = self.world.spawn_actor(cam_bp_depth, cam_trans, attach_to=self.ego_vehicle)
        self.camera_depth.listen(self.depth_queue.put)
        self.actors_to_destroy.append(self.camera_depth)
        
        inst_bp = bp_lib.find('sensor.camera.instance_segmentation')
        inst_bp.set_attribute('image_size_x', str(self.width))
        inst_bp.set_attribute('image_size_y', str(self.height))
        inst_bp.set_attribute('fov', str(fov))
        self.inst_camera = self.world.spawn_actor(inst_bp, cam_trans, attach_to=self.ego_vehicle)
        self.inst_camera.listen(self.inst_queue.put)
        self.actors_to_destroy.append(self.inst_camera)

        # 3. NPC 车辆 (去除自行车)
        vehicle_bps = [x for x in bp_lib.filter('vehicle.*') if int(x.get_attribute('number_of_wheels')) == 4]
        for _ in range(self.num_vehicles):
            if not spawn_points: break
            bp = random.choice(vehicle_bps)
            t = random.choice(spawn_points)
            spawn_points.remove(t)
            npc = self.world.try_spawn_actor(bp, t)
            if npc:
                npc.set_autopilot(True, self.traffic_manager.get_port())
                self.actors_to_destroy.append(npc)

        # 4. NPC 行人 (修正后的批量生成逻辑)
        logging.info("Spawning Walkers...")
        walker_bps = bp_lib.filter('walker.pedestrian.*')
        
        # A. 生成位置
        spawn_points_w = []
        for i in range(self.num_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points_w.append(spawn_point)

        # B. 批量生成行人实体
        batch = []
        for spawn_point in spawn_points_w:
            walker_bp = random.choice(walker_bps)
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            batch.append(carla.command.SpawnActor(walker_bp, spawn_point))
        
        results = self.client.apply_batch_sync(batch, True)
        
        walkers_list = []
        for i in range(len(results)):
            if not results[i].error:
                walkers_list.append({"id": results[i].actor_id})

        # C. 批量生成控制器
        batch = []
        walker_controller_bp = bp_lib.find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(carla.command.SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if not results[i].error:
                walkers_list[i]["con"] = results[i].actor_id
        
        # D. 保存引用并启动
        for i in range(len(walkers_list)):
            actor_id = walkers_list[i]["id"]
            con_id = walkers_list[i]["con"]
            
            walker = self.world.get_actor(actor_id)
            con = self.world.get_actor(con_id)
            
            self.actors_to_destroy.append(walker)
            self.actors_to_destroy.append(con)
            self.walker_controllers.append((con, walker))
            
            # 关键：初始化控制
            con.start()
            con.go_to_location(self.world.get_random_location_from_navigation())
            con.set_max_speed(1.2 + random.random())

    def _trigger_jaywalking(self):
        ego_loc = self.ego_vehicle.get_location()
        candidates = []
        for con, walker in self.walker_controllers:
            if not walker.is_alive or not con.is_alive: continue
            dist = walker.get_location().distance(ego_loc)
            if 10.0 < dist < 30.0: # 距离近一点
                candidates.append((con, walker))
        
        if not candidates: return
        
        con, walker = random.choice(candidates)
        con.stop() # 暂停 AI
        
        # 简单冲向车辆前方
        vec = ego_loc - walker.get_location()
        vec.z = 0
        norm = math.sqrt(vec.x**2 + vec.y**2)
        if norm > 0:
            vec.x /= norm
            vec.y /= norm
            
        control = carla.WalkerControl()
        control.speed = 3.0
        control.direction = vec
        control.jump = False
        walker.apply_control(control)
        # 注意：这里我们让它一直跑，不恢复AI，作为干扰项

    def _game_loop(self, duration=None):
        start_time = time.time()
        target_speed_kmh = 30.0
        
        while self.run_simulation:
            if duration is not None and (time.time() - start_time) >= duration: break
            self._process_events()
            
            # --- 控制逻辑 ---
            # 1. 寻找最近的路径点
            ego_loc = self.ego_vehicle.get_location()
            current_waypoint = self.map.get_waypoint(ego_loc)
            
            # 2. 预瞄前方5米的点
            next_wps = current_waypoint.next(5.0)
            if next_wps:
                target_wp = next_wps[0]
                
                # 3. 运行控制器 (带安全感知)
                control, status = self.controller.run_step(self.ego_vehicle, self.world, target_wp, target_speed_kmh)
                self.ego_vehicle.apply_control(control)
                
                # 状态显示
                status_text = f"Status: {status} | Target: {target_speed_kmh}km/h"
            else:
                self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0, brake=1))
                status_text = "Status: End of Road"

            # --- Jaywalking ---
            self.jaywalking_timer += 1
            if self.jaywalking_timer > TARGET_FPS * 8: # 每8秒尝试一次
                self.jaywalking_timer = 0
                if random.random() < 0.5:
                    self._trigger_jaywalking()

            self.world.tick()

            # --- 渲染 ---
            try:
                img_rgb = self.rgb_queue.get(timeout=1.0)
                img_depth = self.depth_queue.get(timeout=1.0)
                img_inst = self.inst_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            rgb_arr = draw_carla_image(self.display, np.reshape(np.copy(img_rgb.raw_data), (self.height, self.width, 4)))
            depth_map = process_depth_image(img_depth)
            sem, ids = decode_instance_segmentation(np.reshape(np.copy(img_inst.raw_data), (self.height, self.width, 4)))
            
            yolo_res = self.model.track(rgb_arr, persist=True, verbose=False)
            if yolo_res: self.last_inference_time = yolo_res[0].speed.get('inference', 0.0)
            
            w2c = np.array(self.camera_rgb.get_transform().get_inverse_matrix())
            K = build_projection_matrix(self.width, self.height, 70.0)
            
            draw_yolo_and_truth(self.display, self.yolo_font, yolo_res, self.model, depth_map, ids, self.world, 
                                self.ego_vehicle, self.camera_bp_rgb, self.camera_rgb, K, w2c, self.vehicle_tracker)
            
            # 绘制控制状态
            color = (0, 255, 0) if "Normal" in status_text else (255, 0, 0)
            text_surf = self.yolo_font.render(status_text, True, color)
            self.display.blit(text_surf, (10, 40))
            
            draw_fps_info(self.display, self.yolo_font, self.last_inference_time)
            pygame.display.flip()
            self.clock.tick(TARGET_FPS)

    def run(self, duration=None):
        try:
            self._init_pygame()
            self._load_yolo_model()
            self._setup_carla()
            self._spawn_actors()
            self._game_loop(duration)
        except KeyboardInterrupt: pass
        except Exception as e: logging.error(f"{e}", exc_info=True)
        finally: self.cleanup()

    def _process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYUP and event.key == K_ESCAPE):
                self.run_simulation = False

    def cleanup(self):
        if self.world:
            s = self.world.get_settings()
            s.synchronous_mode = False
            self.world.apply_settings(s)
        if self.client:
            for actor in self.actors_to_destroy:
                if hasattr(actor, 'is_listening') and actor.is_listening: actor.stop()
                if hasattr(actor, 'is_alive') and actor.is_alive: actor.destroy()
        pygame.quit()