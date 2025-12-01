# Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


"""
全局路径规划器模块

该模块实现了CARLA仿真环境中的全局路径规划功能，使用A*算法在构建的拓扑图上搜索最短路径。
主要功能包括：
1. 构建地图拓扑结构
2. 创建基于networkx的有向图表示
3. 实现A*路径搜索算法
4. 生成包含道路选项（如直行、左转、右转、变道）的完整路径
5. 支持车道变换连接

使用示例：
    planner = GlobalRoutePlanner(map, sampling_resolution)
    route = planner.trace_route(origin_location, destination_location)
"""
import sys
import os
# 添加外部依赖路径
try:
    sys.path.append(os.path.join(
                    os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))), 'official'))
    sys.path.append(os.path.join(
                    os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))), 'utils'))
except IndexError:
    pass

import math
import numpy as np
import networkx as nx

import carla
from local_planner import RoadOption  # 道路选项枚举
from misc import vector  # 向量计算工具


class GlobalRoutePlanner(object):
    """
    全局路径规划器类
    
    该类负责构建地图的拓扑图表示，并使用A*算法搜索从起点到终点的最短路径。
    路径规划结果包含一系列路点和对应的道路选项（如直行、左转、右转、变道等）。
    """

    def __init__(self, wmap, sampling_resolution):
        """
        初始化全局路径规划器
        
        Args:
            wmap: carla.Map对象，用于获取地图信息
            sampling_resolution: 路径点采样分辨率（米）
        """
        self._sampling_resolution = sampling_resolution  # 路径采样分辨率
        self._wmap = wmap  # 地图对象
        self._topology = None  # 拓扑结构列表
        self._graph = None  # networkx有向图
        self._id_map = None  # 坐标到节点ID的映射 {(x,y,z): id}
        self._road_id_to_edge = None  # 道路ID到边的映射 {road_id: {section_id: {lane_id: (n1, n2)}}}

        self._intersection_end_node = -1  # 交叉路口结束节点ID
        self._previous_decision = RoadOption.VOID  # 上一个道路决策

        # 构建图结构
        self._build_topology()  # 构建拓扑结构
        self._build_graph()  # 构建图
        self._find_loose_ends()  # 处理松散端点
        self._lane_change_link()  # 添加车道变换连接

    def trace_route(self, origin, destination):
        """
        生成从起点到终点的完整路径
        
        Args:
            origin: carla.Location对象，起点位置
            destination: carla.Location对象，终点位置
            
        Returns:
            list: 路径点列表，每个元素为(carla.Waypoint, RoadOption)元组
                 包含路点和对应的道路选项
        """
        route_trace = []  # 最终路径结果
        route = self._path_search(origin, destination)  # 搜索图中的节点路径
        current_waypoint = self._wmap.get_waypoint(origin)  # 当前路点
        destination_waypoint = self._wmap.get_waypoint(destination)  # 终点路点

        # 遍历路径中的每一段边
        for i in range(len(route) - 1):
            road_option = self._turn_decision(i, route)  # 计算道路选项
            edge = self._graph.edges[route[i], route[i+1]]  # 获取当前边
            path = []

            # 处理非车道跟随的边（如变道）
            if edge['type'] != RoadOption.LANEFOLLOW and edge['type'] != RoadOption.VOID:
                route_trace.append((current_waypoint, road_option))
                exit_wp = edge['exit_waypoint']
                # 获取下一条边
                n1, n2 = self._road_id_to_edge[exit_wp.road_id][exit_wp.section_id][exit_wp.lane_id]
                next_edge = self._graph.edges[n1, n2]
                
                # 找到下一条边中最近的路点
                if next_edge['path']:
                    closest_index = self._find_closest_in_list(current_waypoint, next_edge['path'])
                    closest_index = min(len(next_edge['path'])-1, closest_index+5)  # 向前偏移5个点
                    current_waypoint = next_edge['path'][closest_index]
                else:
                    current_waypoint = next_edge['exit_waypoint']
                route_trace.append((current_waypoint, road_option))

            # 处理车道跟随的边
            else:
                # 构建完整路径
                path = path + [edge['entry_waypoint']] + edge['path'] + [edge['exit_waypoint']]
                # 找到当前位置在路径中的最近点
                closest_index = self._find_closest_in_list(current_waypoint, path)
                
                # 遍历路径点，添加到结果中
                for waypoint in path[closest_index:]:
                    current_waypoint = waypoint
                    route_trace.append((current_waypoint, road_option))
                    
                    # 接近终点时提前终止
                    if len(route)-i <= 2 and waypoint.transform.location.distance(destination) < 2*self._sampling_resolution:
                        break
                    # 到达终点车道时终止
                    elif len(route)-i <= 2 and current_waypoint.road_id == destination_waypoint.road_id and current_waypoint.section_id == destination_waypoint.section_id and current_waypoint.lane_id == destination_waypoint.lane_id:
                        destination_index = self._find_closest_in_list(destination_waypoint, path)
                        if closest_index > destination_index:
                            break

        return route_trace

    def _build_topology(self):
        """
        构建地图拓扑结构
        
        从CARLA服务器获取道路段信息，并处理为拓扑结构列表。每个拓扑段包含：
        - 入口路点和出口路点
        - 入口和出口的坐标
        - 入口到出口之间的路点列表
        """
        self._topology = []
        # 从地图获取拓扑结构（道路段列表）
        for segment in self._wmap.get_topology():
            wp1, wp2 = segment[0], segment[1]  # 道路段的起点和终点路点
            l1, l2 = wp1.transform.location, wp2.transform.location  # 起点和终点位置
            
            # 四舍五入坐标，避免浮点数精度问题
            x1, y1, z1, x2, y2, z2 = np.round([l1.x, l1.y, l1.z, l2.x, l2.y, l2.z], 0)
            wp1.transform.location, wp2.transform.location = l1, l2  # 恢复原始位置
            
            # 构建道路段字典
            seg_dict = dict()
            seg_dict['entry'], seg_dict['exit'] = wp1, wp2  # 入口和出口路点
            seg_dict['entryxyz'], seg_dict['exitxyz'] = (x1, y1, z1), (x2, y2, z2)  # 入口和出口坐标
            seg_dict['path'] = []  # 路径点列表
            
            endloc = wp2.transform.location
            # 如果起点到终点的距离大于采样分辨率，则生成中间路点
            if wp1.transform.location.distance(endloc) > self._sampling_resolution:
                w = wp1.next(self._sampling_resolution)[0]  # 获取下一个路点
                # 循环生成中间路点，直到接近终点
                while w.transform.location.distance(endloc) > self._sampling_resolution:
                    seg_dict['path'].append(w)
                    w = w.next(self._sampling_resolution)[0]
            else:
                # 距离较近时，至少添加一个中间路点
                seg_dict['path'].append(wp1.next(self._sampling_resolution)[0])
            
            self._topology.append(seg_dict)

    def _build_graph(self):
        """
        构建基于networkx的有向图表示
        
        创建以下类属性：
        - _graph: networkx.DiGraph对象，节点表示位置，边表示道路段
        - _id_map: 坐标到节点ID的映射
        - _road_id_to_edge: 道路ID到图中边的映射
        
        图的属性：
        - 节点属性：vertex - 世界坐标(x,y,z)
        - 边属性：
            length: 边的长度（路点数量）
            path: 边包含的路点列表
            entry_waypoint: 入口路点
            exit_waypoint: 出口路点
            entry_vector: 入口处的单位切向量
            exit_vector: 出口处的单位切向量
            net_vector: 入口到出口的单位向量
            intersection: 是否属于交叉路口
            type: 道路类型（RoadOption）
        """
        # 初始化图和映射字典
        self._graph = nx.DiGraph()
        self._id_map = dict()  # 结构：{(x,y,z): id, ... }
        self._road_id_to_edge = dict()  # 结构：{road_id: {section_id: {lane_id: edge, ... }, ... }}

        # 遍历拓扑结构，构建图
        for segment in self._topology:
            entry_xyz, exit_xyz = segment['entryxyz'], segment['exitxyz']
            path = segment['path']
            entry_wp, exit_wp = segment['entry'], segment['exit']
            intersection = entry_wp.is_junction  # 是否属于交叉路口
            road_id, section_id, lane_id = entry_wp.road_id, entry_wp.section_id, entry_wp.lane_id

            # 添加节点到图中
            for vertex in entry_xyz, exit_xyz:
                # 仅添加未存在的节点
                if vertex not in self._id_map:
                    new_id = len(self._id_map)  # 分配新ID
                    self._id_map[vertex] = new_id
                    self._graph.add_node(new_id, vertex=vertex)  # 添加节点，包含坐标属性
            
            # 获取节点ID
            n1 = self._id_map[entry_xyz]  # 入口节点ID
            n2 = self._id_map[exit_xyz]  # 出口节点ID
            
            # 更新road_id到edge的映射
            if road_id not in self._road_id_to_edge:
                self._road_id_to_edge[road_id] = dict()
            if section_id not in self._road_id_to_edge[road_id]:
                self._road_id_to_edge[road_id][section_id] = dict()
            self._road_id_to_edge[road_id][section_id][lane_id] = (n1, n2)

            # 计算向量属性
            entry_carla_vector = entry_wp.transform.rotation.get_forward_vector()  # 入口前向向量
            exit_carla_vector = exit_wp.transform.rotation.get_forward_vector()  # 出口前向向量

            # 添加边到图中
            self._graph.add_edge(
                n1, n2,
                length=len(path) + 1,  # 边的长度（路点数量）
                path=path,  # 边包含的路点列表
                entry_waypoint=entry_wp,  # 入口路点
                exit_waypoint=exit_wp,  # 出口路点
                entry_vector=np.array([entry_carla_vector.x, entry_carla_vector.y, entry_carla_vector.z]),  # 入口单位切向量
                exit_vector=np.array([exit_carla_vector.x, exit_carla_vector.y, exit_carla_vector.z]),  # 出口单位切向量
                net_vector=vector(entry_wp.transform.location, exit_wp.transform.location),  # 入口到出口的单位向量
                intersection=intersection,  # 是否属于交叉路口
                type=RoadOption.LANEFOLLOW)  # 道路类型（默认车道跟随）

    def _find_loose_ends(self):
        """
        查找并处理松散端点
        
        松散端点是指道路段的未连接端，该方法会将这些端点添加到图中，
        确保图的完整性，避免路径搜索时出现断路。
        """
        count_loose_ends = 0  # 松散端点计数
        hop_resolution = self._sampling_resolution  # 跳跃分辨率
        
        # 遍历拓扑结构
        for segment in self._topology:
            end_wp = segment['exit']  # 道路段的出口路点
            exit_xyz = segment['exitxyz']  # 出口坐标
            road_id, section_id, lane_id = end_wp.road_id, end_wp.section_id, end_wp.lane_id
            
            # 检查该道路段是否已连接
            if road_id in self._road_id_to_edge \
                    and section_id in self._road_id_to_edge[road_id] \
                    and lane_id in self._road_id_to_edge[road_id][section_id]:
                continue  # 已连接，跳过
            else:
                count_loose_ends += 1  # 松散端点计数+1
                
                # 更新road_id到edge的映射
                if road_id not in self._road_id_to_edge:
                    self._road_id_to_edge[road_id] = dict()
                if section_id not in self._road_id_to_edge[road_id]:
                    self._road_id_to_edge[road_id][section_id] = dict()
                
                # 创建新的边
                n1 = self._id_map[exit_xyz]  # 起始节点ID
                n2 = -1 * count_loose_ends  # 分配负ID作为松散端点
                self._road_id_to_edge[road_id][section_id][lane_id] = (n1, n2)
                
                # 生成松散端点的路径
                next_wp = end_wp.next(hop_resolution)  # 获取下一个路点
                path = []
                # 循环生成路径，直到路点不再属于当前道路段
                while next_wp is not None and next_wp \
                        and next_wp[0].road_id == road_id \
                        and next_wp[0].section_id == section_id \
                        and next_wp[0].lane_id == lane_id:
                    path.append(next_wp[0])
                    next_wp = next_wp[0].next(hop_resolution)
                
                # 如果生成了路径，添加到图中
                if path:
                    # 获取路径终点坐标
                    n2_xyz = (path[-1].transform.location.x,
                              path[-1].transform.location.y,
                              path[-1].transform.location.z)
                    # 添加节点和边
                    self._graph.add_node(n2, vertex=n2_xyz)
                    self._graph.add_edge(
                        n1, n2,
                        length=len(path) + 1, path=path,
                        entry_waypoint=end_wp, exit_waypoint=path[-1],
                        entry_vector=None, exit_vector=None, net_vector=None,
                        intersection=end_wp.is_junction, type=RoadOption.LANEFOLLOW)

    def _lane_change_link(self):
        """
        添加车道变换连接
        
        在拓扑图中添加零成本的边，表示允许的车道变换。
        这些边连接相邻车道，允许路径规划器在需要时进行车道变换。
        """
        # 遍历拓扑结构
        for segment in self._topology:
            left_found, right_found = False, False  # 标记是否已找到左/右变道连接

            # 遍历道路段的路点
            for waypoint in segment['path']:
                if not segment['entry'].is_junction:  # 非交叉路口才允许变道
                    next_waypoint, next_road_option, next_segment = None, None, None

                    # 检查右侧车道变换
                    if waypoint.right_lane_marking and waypoint.right_lane_marking.lane_change & carla.LaneChange.Right and not right_found:
                        next_waypoint = waypoint.get_right_lane()  # 获取右侧车道
                        # 检查右侧车道是否可用
                        if next_waypoint is not None \
                                and next_waypoint.lane_type == carla.LaneType.Driving \
                                and waypoint.road_id == next_waypoint.road_id:
                            next_road_option = RoadOption.CHANGELANERIGHT  # 右侧变道
                            next_segment = self._localize(next_waypoint.transform.location)  # 定位右侧车道
                            if next_segment is not None:
                                # 添加右侧变道边
                                self._graph.add_edge(
                                    self._id_map[segment['entryxyz']], next_segment[0], 
                                    entry_waypoint=waypoint,
                                    exit_waypoint=next_waypoint, 
                                    intersection=False, 
                                    exit_vector=None,
                                    path=[], 
                                    length=0,  # 变道成本为0
                                    type=next_road_option, 
                                    change_waypoint=next_waypoint)
                                right_found = True  # 标记已找到右侧变道
                    
                    # 检查左侧车道变换
                    if waypoint.left_lane_marking and waypoint.left_lane_marking.lane_change & carla.LaneChange.Left and not left_found:
                        next_waypoint = waypoint.get_left_lane()  # 获取左侧车道
                        # 检查左侧车道是否可用
                        if next_waypoint is not None \
                                and next_waypoint.lane_type == carla.LaneType.Driving \
                                and waypoint.road_id == next_waypoint.road_id:
                            next_road_option = RoadOption.CHANGELANELEFT  # 左侧变道
                            next_segment = self._localize(next_waypoint.transform.location)  # 定位左侧车道
                            if next_segment is not None:
                                # 添加左侧变道边
                                self._graph.add_edge(
                                    self._id_map[segment['entryxyz']], next_segment[0], 
                                    entry_waypoint=waypoint,
                                    exit_waypoint=next_waypoint, 
                                    intersection=False, 
                                    exit_vector=None,
                                    path=[], 
                                    length=0,  # 变道成本为0
                                    type=next_road_option, 
                                    change_waypoint=next_waypoint)
                                left_found = True  # 标记已找到左侧变道
                
                # 左右变道都找到后，跳出循环
                if left_found and right_found:
                    break

    def _localize(self, location):
        """
        定位位置所在的道路段
        
        Args:
            location: carla.Location对象，要定位的位置
            
        Returns:
            tuple: 包含边的起点和终点节点ID的元组，或None
        """
        waypoint = self._wmap.get_waypoint(location)  # 获取该位置的路点
        edge = None
        try:
            # 通过road_id, section_id, lane_id查找边
            edge = self._road_id_to_edge[waypoint.road_id][waypoint.section_id][waypoint.lane_id]
        except KeyError:
            pass  # 未找到对应边
        return edge

    def _distance_heuristic(self, n1, n2):
        """
        距离启发式函数
        
        A*算法使用的启发式函数，计算两个节点之间的欧几里得距离。
        
        Args:
            n1: 节点1的ID
            n2: 节点2的ID
            
        Returns:
            float: 两个节点之间的欧几里得距离
        """
        l1 = np.array(self._graph.nodes[n1]['vertex'])  # 节点1的坐标
        l2 = np.array(self._graph.nodes[n2]['vertex'])  # 节点2的坐标
        return np.linalg.norm(l1-l2)  # 欧几里得距离

    def _path_search(self, origin, destination):
        """
        路径搜索
        
        使用A*算法搜索从起点到终点的最短路径。
        
        Args:
            origin: carla.Location对象，起点位置
            destination: carla.Location对象，终点位置
            
        Returns:
            list: 路径节点ID列表
        """
        # 定位起点和终点所在的边
        start, end = self._localize(origin), self._localize(destination)

        # 使用A*算法搜索路径
        route = nx.astar_path(
            self._graph, 
            source=start[0],  # 起点边的起始节点
            target=end[0],  # 终点边的起始节点
            heuristic=self._distance_heuristic,  # 距离启发式函数
            weight='length')  # 边权重（路点数量）
        route.append(end[1])  # 添加终点边的终点节点
        return route

    def _successive_last_intersection_edge(self, index, route):
        """
        获取连续的最后交叉路口边
        
        该方法用于跳过小的交叉路口边，找到真正的交叉路口出口边，
        以便正确计算转弯决策。
        
        Args:
            index: 路径中的当前索引
            route: 路径节点ID列表
            
        Returns:
            tuple: (最后交叉路口节点ID, 最后交叉路口边)
        """
        last_intersection_edge = None  # 最后交叉路口边
        last_node = None  # 最后交叉路口节点
        
        # 遍历从当前索引开始的边
        for node1, node2 in [(route[i], route[i+1]) for i in range(index, len(route)-1)]:
            candidate_edge = self._graph.edges[node1, node2]  # 当前边
            
            if node1 == route[index]:
                last_intersection_edge = candidate_edge  # 初始化为当前边
            
            # 如果是车道跟随且属于交叉路口
            if candidate_edge['type'] == RoadOption.LANEFOLLOW and candidate_edge['intersection']:
                last_intersection_edge = candidate_edge  # 更新为当前边
                last_node = node2  # 更新最后节点
            else:
                break  # 非交叉路口边，跳出循环

        return last_node, last_intersection_edge

    def _turn_decision(self, index, route, threshold=math.radians(35)):
        """
        计算转弯决策
        
        根据当前边和下一条边的关系，计算道路选项（如直行、左转、右转）。
        
        Args:
            index: 路径中的当前索引
            route: 路径节点ID列表
            threshold: 转弯角度阈值（弧度），小于该值视为直行
            
        Returns:
            RoadOption: 道路选项（如STRAIGHT, LEFT, RIGHT, LANEFOLLOW等）
        """
        decision = None  # 决策结果
        previous_node = route[index-1]  # 前一个节点
        current_node = route[index]  # 当前节点
        next_node = route[index+1]  # 下一个节点
        next_edge = self._graph.edges[current_node, next_node]  # 下一条边
        
        if index > 0:  # 不是路径的第一个节点
            # 检查是否需要延续之前的决策
            if self._previous_decision != RoadOption.VOID \
                    and self._intersection_end_node > 0 \
                    and self._intersection_end_node != previous_node \
                    and next_edge['type'] == RoadOption.LANEFOLLOW \
                    and next_edge['intersection']:
                decision = self._previous_decision  # 延续之前的决策
            else:
                self._intersection_end_node = -1  # 重置交叉路口结束节点
                current_edge = self._graph.edges[previous_node, current_node]  # 当前边
                
                # 检查是否需要计算转弯决策
                calculate_turn = current_edge['type'] == RoadOption.LANEFOLLOW and not current_edge[
                    'intersection'] and next_edge['type'] == RoadOption.LANEFOLLOW and next_edge['intersection']
                
                if calculate_turn:
                    # 获取连续的最后交叉路口边
                    last_node, tail_edge = self._successive_last_intersection_edge(index, route)
                    self._intersection_end_node = last_node  # 更新交叉路口结束节点
                    if tail_edge is not None:
                        next_edge = tail_edge  # 使用最后交叉路口边
                    
                    # 获取当前边的出口向量和下一条边的出口向量
                    cv, nv = current_edge['exit_vector'], next_edge['exit_vector']
                    if cv is None or nv is None:
                        return next_edge['type']  # 向量无效时，返回下一条边的类型
                    
                    # 计算交叉乘积，用于判断转弯方向
                    cross_list = []  # 存储其他可能路径的交叉乘积
                    for neighbor in self._graph.successors(current_node):  # 遍历当前节点的所有邻居
                        select_edge = self._graph.edges[current_node, neighbor]  # 邻居边
                        if select_edge['type'] == RoadOption.LANEFOLLOW:  # 仅考虑车道跟随边
                            if neighbor != route[index+1]:  # 排除当前选择的路径
                                sv = select_edge['net_vector']  # 邻居边的净向量
                                cross_list.append(np.cross(cv, sv)[2])  # 计算交叉乘积的z分量
                    
                    next_cross = np.cross(cv, nv)[2]  # 当前选择路径的交叉乘积
                    # 计算向量夹角（弧度）
                    deviation = math.acos(np.clip(
                        np.dot(cv, nv)/(np.linalg.norm(cv)*np.linalg.norm(nv)), -1.0, 1.0))
                    
                    if not cross_list:
                        cross_list.append(0)  # 避免空列表
                    
                    # 根据夹角和交叉乘积判断转弯方向
                    if deviation < threshold:  # 夹角小于阈值，视为直行
                        decision = RoadOption.STRAIGHT
                    elif cross_list and next_cross < min(cross_list):  # 交叉乘积小于所有其他路径，左转
                        decision = RoadOption.LEFT
                    elif cross_list and next_cross > max(cross_list):  # 交叉乘积大于所有其他路径，右转
                        decision = RoadOption.RIGHT
                    elif next_cross < 0:  # 交叉乘积为负，左转
                        decision = RoadOption.LEFT
                    elif next_cross > 0:  # 交叉乘积为正，右转
                        decision = RoadOption.RIGHT
                else:
                    decision = next_edge['type']  # 不需要计算转弯，直接使用下一条边的类型

        else:  # 路径的第一个节点
            decision = next_edge['type']  # 直接使用下一条边的类型

        self._previous_decision = decision  # 保存当前决策
        return decision

    def _find_closest_in_list(self, current_waypoint, waypoint_list):
        """
        查找列表中最近的路点
        
        Args:
            current_waypoint: 当前路点
            waypoint_list: 路点列表
            
        Returns:
            int: 最近路点在列表中的索引
        """
        min_distance = float('inf')  # 最小距离（初始化为无穷大）
        closest_index = -1  # 最近路点索引
        
        # 遍历路点列表
        for i, waypoint in enumerate(waypoint_list):
            # 计算当前路点与列表中路点的距离
            distance = waypoint.transform.location.distance(
                current_waypoint.transform.location)
            # 更新最小距离和最近索引
            if distance < min_distance:
                min_distance = distance
                closest_index = i

        return closest_index