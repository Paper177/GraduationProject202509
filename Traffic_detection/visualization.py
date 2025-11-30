import pygame
import numpy as np
import math
import carla
import utils
import traffic_light  # 引入新模块

def draw_traffic_lights(surface, font, yolo_results, yolo_model, img_rgb, depth_map, detection_method='deep_learning'):
    """
    专门用于绘制红绿灯及其颜色状态，支持深度学习方法和YOLO直接分类功能，并输出相对位置
    
    Args:
        surface: Pygame显示表面
        font: 字体对象
        yolo_results: YOLO检测结果
        yolo_model: YOLO模型对象
        img_rgb: RGB图像数组
        depth_map: 深度图，用于计算距离
        detection_method: 颜色检测方法，可选 'deep_learning' 或 'hsv'
    """
    # 获取图像中心，用于计算相对位置
    img_center_x = surface.get_width() // 2
    
    # 首先尝试使用YOLO直接分类功能
    tl_results = traffic_light.register_yolo_tl_classification(yolo_results, yolo_model.names)
    
    # 处理有直接分类结果的红绿灯
    for bbox_tuple, result in tl_results.items():
        x1, y1, x2, y2 = bbox_tuple
        color_str = result['color']
        conf = result['conf']
        
        # 如果YOLO没有直接给出颜色，使用我们的颜色检测方法
        if color_str == 'UNKNOWN' and detection_method:
            # 1. 截取红绿灯区域图片
            roi_rgb = img_rgb[y1:y2, x1:x2]
            # 2. 识别红绿灯颜色
            color_str, box_color = traffic_light.get_traffic_light_color(
                img_rgb, [x1, y1, x2, y2], method=detection_method
            )
        else:
            box_color = traffic_light.COLOR_MAP.get(color_str, traffic_light.COLOR_MAP['UNKNOWN'])
        
        w = x2 - x1
        h = y2 - y1
        
        # 计算红绿灯中心位置
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        # 计算相对车辆的位置
        # 1. 距离计算
        roi_depth = depth_map[y1:y2, x1:x2]
        z_values = roi_depth[roi_depth > 0]
        if len(z_values) > 0:
            distance = np.median(z_values)
        else:
            distance = -1.0
        
        # 2. 水平方向相对位置
        if cx < img_center_x - 100:
            relative_pos = "Left"
        elif cx > img_center_x + 100:
            relative_pos = "Right"
        else:
            relative_pos = "Center"
        
        # 3. 垂直方向相对位置
        if cy < surface.get_height() // 3:
            vertical_pos = "High"
        elif cy > surface.get_height() * 2 // 3:
            vertical_pos = "Low"
        else:
            vertical_pos = "Middle"
        
        # 绘制框
        pygame.draw.rect(surface, box_color, pygame.Rect(x1, y1, w, h), 2)
        
        # 绘制详细标签
        label = f"TL: {color_str} ({conf:.2f})"
        pos_label = f"Pos: {relative_pos} {vertical_pos}"
        dist_label = f"Dist: {distance:.1f} m" if distance > 0 else "Dist: Unknown"
        
        text_surface = font.render(label, True, (255, 255, 255), box_color)
        pos_text_surface = font.render(pos_label, True, (255, 255, 255), box_color)
        dist_text_surface = font.render(dist_label, True, (255, 255, 255), box_color)
        
        # 绘制标签，垂直排列
        text_rect = text_surface.get_rect(bottomleft=(x1, y1 - 25)) 
        pos_text_rect = pos_text_surface.get_rect(bottomleft=(x1, text_rect.top - 5))
        dist_text_rect = dist_text_surface.get_rect(bottomleft=(x1, pos_text_rect.top - 5))
        
        surface.blit(dist_text_surface, dist_text_rect)
        surface.blit(pos_text_surface, pos_text_rect)
        surface.blit(text_surface, text_rect)
    
    # 遍历剩余的检测结果，确保没有遗漏任何红绿灯
    for r in yolo_results:
        # 获取所有框的数据
        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        class_ids = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy().astype(float)

        for box, class_id, conf in zip(boxes, class_ids, confs):
            x1, y1, x2, y2 = box
            
            # 检查此红绿灯是否已经处理过
            if (x1, y1, x2, y2) in tl_results:
                continue
                
            # 1. 类别过滤
            class_name = yolo_model.names.get(class_id)
            if class_name != traffic_light.YOLO_TARGET_CLASS:
                continue
                
            if conf < 0.5: # 简单的置信度过滤
                continue
            
            # 2. 截取红绿灯区域图片
            roi_rgb = img_rgb[y1:y2, x1:x2]
            # 3. 识别红绿灯颜色
            color_str, box_color = traffic_light.get_traffic_light_color(img_rgb, box, method=detection_method)
            
            # 4. 计算相对位置
            # 计算红绿灯中心位置
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            # 距离计算
            roi_depth = depth_map[y1:y2, x1:x2]
            z_values = roi_depth[roi_depth > 0]
            if len(z_values) > 0:
                distance = np.median(z_values)
            else:
                distance = -1.0
            
            # 水平方向相对位置
            if cx < img_center_x - 100:
                relative_pos = "Left"
            elif cx > img_center_x + 100:
                relative_pos = "Right"
            else:
                relative_pos = "Center"
            
            # 垂直方向相对位置
            if cy < surface.get_height() // 3:
                vertical_pos = "High"
            elif cy > surface.get_height() * 2 // 3:
                vertical_pos = "Low"
            else:
                vertical_pos = "Middle"
            
            # 5. 绘制
            w = x2 - x1
            h = y2 - y1
            
            # 绘制框
            pygame.draw.rect(surface, box_color, pygame.Rect(x1, y1, w, h), 2)
            
            # 绘制详细标签
            label = f"TL: {color_str} ({conf:.2f})"
            pos_label = f"Pos: {relative_pos} {vertical_pos}"
            dist_label = f"Dist: {distance:.1f} m" if distance > 0 else "Dist: Unknown"
            
            text_surface = font.render(label, True, (255, 255, 255), box_color)
            pos_text_surface = font.render(pos_label, True, (255, 255, 255), box_color)
            dist_text_surface = font.render(dist_label, True, (255, 255, 255), box_color)
            
            # 绘制标签，垂直排列
            text_rect = text_surface.get_rect(bottomleft=(x1, y1 - 25)) 
            pos_text_rect = pos_text_surface.get_rect(bottomleft=(x1, text_rect.top - 5))
            dist_text_rect = dist_text_surface.get_rect(bottomleft=(x1, pos_text_rect.top - 5))
            
            surface.blit(dist_text_surface, dist_text_rect)
            surface.blit(pos_text_surface, pos_text_rect)
            surface.blit(text_surface, text_rect)

def draw_vehicles_and_truth(
        surface, font, yolo_results, yolo_model, 
        depth_map, # 算法深度图
        actor_ids_map, # 真值 ID 查找表
        world, ego_vehicle, 
        camera_bp, camera_left, K_left, world_2_camera_left, 
        vehicle_tracker # 卡尔曼跟踪器
    ):
    """
    (原 draw_yolo_and_truth，重命名以区分功能)
    负责车辆检测、测距、测速和真值对比
    """
    
    drawn_actor_ids = set()

    for r in yolo_results:
        if r.boxes.id is None:
            continue

        boxes = r.boxes.xyxy.cpu().numpy().astype(int)
        track_ids = r.boxes.id.cpu().numpy().astype(int)
        class_ids = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy().astype(float)

        for box, track_id, class_id, conf in zip(boxes, track_ids, class_ids, confs):
            
            x1, y1, x2, y2 = box
            class_name = yolo_model.names.get(class_id)
            
            if class_name not in utils.YOLO_TARGET_CLASSES:
                continue
            
            # --- 算法：测量距离 (Z) ---
            roi = depth_map[y1:y2, x1:x2]
            z_values = roi[roi > 0] # 过滤掉 0
            
            if len(z_values) == 0:
                continue
                
            measured_Z = np.median(z_values)

            # --- 算法：卡尔曼滤波器更新 ---
            kf = vehicle_tracker.get_filter(track_id, measured_Z)
            kf.predict()
            kf.update(np.array([measured_Z]))
            
            computed_Z_smooth = kf.x[0]
            computed_vel_mps = kf.x[1]
            
            # 计算绝对速度
            ego_vel_vec = ego_vehicle.get_velocity()
            ego_speed_mps = math.sqrt(ego_vel_vec.x**2 + ego_vel_vec.y**2 + ego_vel_vec.z**2)
            computed_abs_vel_mps = ego_speed_mps + computed_vel_mps
            computed_vel_kmh = computed_abs_vel_mps * 3.6

            # --- 真值：获取 Actor ID 和速度 (用于对比) ---
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            if not (0 <= cy < actor_ids_map.shape[0] and 0 <= cx < actor_ids_map.shape[1]):
                continue
            
            actor_id = int(actor_ids_map[cy, cx])
            
            if actor_id == 0 or actor_id == ego_vehicle.id or actor_id in drawn_actor_ids:
                continue
            
            actor = world.get_actor(actor_id)
            
            if actor is None or not actor.is_alive or not isinstance(actor, carla.Vehicle):
                continue
                
            drawn_actor_ids.add(actor_id)

            # 获取真值速度
            vel = actor.get_velocity()
            truth_vel_kmh = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            
            # --- 真值：获取 3D 边界框 (用于绘制) ---
            projection = utils.get_truth_3d_box_projection(
                actor, ego_vehicle, camera_bp, 
                camera_left, K_left, world_2_camera_left
            )

            # --- 绘制 3D 边界框 ---
            n = 0
            mean_x = 0
            mean_y = 0
            for line in projection:
                pygame.draw.line(surface, utils.VEHICLE_COLOR, (line[0], line[1]), (line[2],line[3]), 2)
                mean_x += line[0]
                mean_y += line[1]
                n += 1

            # --- 绘制标签 (算法速度 vs 真值速度) ---
            if n > 0:
                mean_x /= n
                mean_y /= n
                
                label_dist = f"Dist: {computed_Z_smooth:.1f} m"
                label_vel = f"Vel: {computed_vel_kmh:.1f} km/h (Truth: {truth_vel_kmh:.1f})"
                
                text_surface_dist = font.render(label_dist, True, (255, 255, 255), utils.VEHICLE_COLOR)
                text_rect_dist = text_surface_dist.get_rect(bottomleft=(mean_x, mean_y - 5))
                surface.blit(text_surface_dist, text_rect_dist)
                
                text_surface_vel = font.render(label_vel, True, (255, 255, 255), utils.VEHICLE_COLOR)
                text_rect_vel = text_surface_vel.get_rect(bottomleft=(mean_x, text_rect_dist.top - 5))
                surface.blit(text_surface_vel, text_rect_vel)