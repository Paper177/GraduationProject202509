import numpy as np

try:
    from filterpy.kalman import KalmanFilter
    from filterpy.common import Q_discrete_white_noise
except ImportError:
    raise RuntimeError('需要 filterpy, 请运行 "pip install filterpy"')

class VehicleTracker(object):
    """
    为每个 track_id 管理一个独立的卡尔曼滤波器，
    以平滑深度相机的距离并估算速度。
    """
    def __init__(self, dt):
        self.dt = dt  # 时间步长 (例如 0.05s)
        self.filters = {}  # {track_id: KalmanFilter}

    def _create_filter(self, initial_z):
        """
        创建新的卡尔曼滤波器:
        状态 x = [z, v_z] (距离, 速度)
        """
        kf = KalmanFilter(dim_x=2, dim_z=1)
        
        # 1. 初始状态
        kf.x = np.array([initial_z, 0.0])
        
        # 2. 状态转移矩阵 F
        kf.F = np.array([[1., self.dt],
                         [0., 1.]])
        
        # 3. 测量函数 H
        kf.H = np.array([[1., 0.]])
        
        # 4. 协方差 P
        kf.P *= 10.0
        
        # 5. 测量噪声 R (深度相机较准，值设小)
        kf.R = np.array([[0.01]])
        
        # 6. 过程噪声 Q
        kf.Q = Q_discrete_white_noise(dim=2, dt=self.dt, var=0.1)
        
        return kf

    def get_filter(self, track_id, computed_Z):
        """获取或创建此 track_id 的滤波器"""
        if track_id not in self.filters:
            self.filters[track_id] = self._create_filter(computed_Z)
        return self.filters[track_id]