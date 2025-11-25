import numpy as np

# 检查 OpenCV 依赖
try:
    import cv2
except ImportError:
    raise RuntimeError('需要 OpenCV, 请运行 "pip install opencv-python"')

# --- 常量定义 ---
YOLO_TARGET_CLASS = 'traffic light'

# HSV 阈值定义 (H: 0-179, S: 0-255, V: 0-255)
# 红色 (环绕两个区间)
RED_LOWER_1 = np.array([0, 120, 120])
RED_UPPER_1 = np.array([10, 255, 255])
RED_LOWER_2 = np.array([170, 120, 120])
RED_UPPER_2 = np.array([179, 255, 255])

# 黄色
YELLOW_LOWER = np.array([20, 120, 120])
YELLOW_UPPER = np.array([35, 255, 255])

# 绿色
GREEN_LOWER = np.array([40, 120, 120])
GREEN_UPPER = np.array([90, 255, 255])

# 颜色映射: 颜色名称 -> RGB 值
COLOR_MAP = {
    "RED": (255, 0, 0),
    "YELLOW": (255, 255, 0),
    "GREEN": (0, 255, 0),
    "UNKNOWN": (128, 128, 128)
}

def get_traffic_light_color(image_rgb, bbox):
    """
    分析图像区域内的颜色，判断红绿灯状态。
    
    Args:
        image_rgb: 完整的 RGB 图像 (numpy array)
        bbox: 边界框 [x1, y1, x2, y2]
        
    Returns:
        tuple: (颜色名称字符串, RGB颜色元组)
    """
    x1, y1, x2, y2 = bbox
    
    # 裁剪 ROI (Region of Interest)
    roi_rgb = image_rgb[y1:y2, x1:x2]

    # 检查有效性
    if roi_rgb.shape[0] == 0 or roi_rgb.shape[1] == 0:
        return "UNKNOWN", COLOR_MAP["UNKNOWN"]

    # 转换到 HSV 空间
    roi_hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)

    # 创建掩码
    mask_red1 = cv2.inRange(roi_hsv, RED_LOWER_1, RED_UPPER_1)
    mask_red2 = cv2.inRange(roi_hsv, RED_LOWER_2, RED_UPPER_2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    mask_yellow = cv2.inRange(roi_hsv, YELLOW_LOWER, YELLOW_UPPER)
    mask_green = cv2.inRange(roi_hsv, GREEN_LOWER, GREEN_UPPER)

    # 统计像素
    pixel_counts = {
        "RED": np.count_nonzero(mask_red),
        "YELLOW": np.count_nonzero(mask_yellow),
        "GREEN": np.count_nonzero(mask_green)
    }

    # 简单的投票机制
    max_color = "UNKNOWN"
    max_pixels = 5 # 噪点阈值
    
    for color, count in pixel_counts.items():
        if count > max_pixels:
            max_pixels = count
            max_color = color
            
    return max_color, COLOR_MAP[max_color]