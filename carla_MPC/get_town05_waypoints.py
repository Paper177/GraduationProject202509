import carla
import time
import json
import os

def main():
    try:
        # 1. 连接到 CARLA 服务器
        print("Connecting to CARLA server...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)

        # 2. 加载 Town05 地图
        print("Loading world: Town05...")
        # 如果当前不是 Town05，则加载它；否则获取当前世界
        world = client.get_world()
        if 'Town05' not in world.get_map().name:
            world = client.load_world('Town05')
        
        map = world.get_map()
        print(f"Map loaded: {map.name}")

        # 3. 获取数据
        spawn_points = map.get_spawn_points()
        distance = 2.0  # Waypoints 间距
        waypoints = map.generate_waypoints(distance)
        
        print(f"Found {len(spawn_points)} spawn points.")
        print(f"Generated {len(waypoints)} waypoints.")

        # 4. 可视化设置
        debug = world.debug
        life_time = 60.0  # 可视化持续时间（秒）
        
        print("Starting visualization...")
        print(f"Markers will remain visible for {life_time} seconds.")

        # --- 可视化 Spawn Points (红色) ---
        # Spawn Points 通常较少，全部标注
        print("Drawing Spawn Points (RED)...")
        for i, sp in enumerate(spawn_points):
            loc = sp.location
            # 画一个较大的红点
            debug.draw_point(loc, size=0.2, color=carla.Color(255, 0, 0), life_time=life_time)
            
            # 绘制箭头指示车头朝向
            # sp 是 carla.Transform 对象，可以直接获取前向向量
            forward_vec = sp.get_forward_vector()
            arrow_begin = loc + carla.Location(z=0.5)
            arrow_end = arrow_begin + forward_vec * 2.0  # 箭头长度 2 米
            debug.draw_arrow(arrow_begin, arrow_end, thickness=0.1, arrow_size=0.1, 
                             color=carla.Color(255, 0, 0), life_time=life_time)

            # 在点上方绘制坐标文本
            # 格式: SP:ID(x, y)
            text = f"SP:{i}\n({loc.x:.1f}, {loc.y:.1f})"
            debug.draw_string(loc + carla.Location(z=1.0), text, 
                            draw_shadow=True, 
                            color=carla.Color(255, 200, 200), 
                            life_time=life_time)

        # --- 可视化 Waypoints (绿色) ---
        # Waypoints 数量巨大，我们画小点，并且只对部分点标注坐标以避免拥挤
        print("Drawing Waypoints (GREEN)...")
        for i, wp in enumerate(waypoints):
            loc = wp.transform.location
            
            # 画一个小绿点
            debug.draw_point(loc, size=0.05, color=carla.Color(0, 255, 0), life_time=life_time)
            
            # 每隔 10 个点 (约20米) 标注一次坐标，避免文字重叠
            if i % 10 == 0:
                text = f"({loc.x:.1f}, {loc.y:.1f})"
                debug.draw_string(loc + carla.Location(z=0.5), text, 
                                draw_shadow=False, 
                                color=carla.Color(0, 255, 0), 
                                life_time=life_time)

        print("Visualization done! Check your CARLA simulator window.")
        
        # 5. 保存数据 (可选，方便查看数值)
        output_file = "town05_waypoints_visualized.json"
        spawn_points_data = [{
            "id": i, "x": sp.location.x, "y": sp.location.y, "z": sp.location.z
        } for i, sp in enumerate(spawn_points)]
        
        with open(output_file, "w") as f:
            json.dump({
                "spawn_points": spawn_points_data,
                "note": "Check CARLA window for visual markers"
            }, f, indent=4)
        print(f"Data saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
