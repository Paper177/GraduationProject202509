import numpy as np

# 检查 OpenCV 依赖
try:
    import cv2
except ImportError:
    raise RuntimeError('需要 OpenCV, 请运行 "pip install opencv-python"')

# 检查 PyTorch 依赖（用于深度学习方法）
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorch 未安装，将使用传统 HSV 方法。可通过 'pip install torch torchvision' 安装。")

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

# 红绿灯颜色类标签
TL_CLASSES = ['RED', 'YELLOW', 'GREEN', 'UNKNOWN']

# 全局变量用于存储训练好的分类模型
_tl_color_model = None
_transform = None

# === 深度学习模型 ===
class SimpleTLClassifier(nn.Module):
    """
    轻量级红绿灯颜色分类CNN模型
    """
    def __init__(self, num_classes=4):
        super(SimpleTLClassifier, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # 激活函数
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # 输入尺寸: [batch_size, 3, 128, 128]
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        # 展开特征图
        x = x.view(-1, 64 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def _init_deep_learning_model():
    """
    初始化深度学习模型和数据转换
    """
    global _tl_color_model, _transform
    
    # 创建模型实例
    _tl_color_model = SimpleTLClassifier(num_classes=len(TL_CLASSES))
    
    # 创建图像转换
    _transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return _tl_color_model, _transform

def get_traffic_light_color_deep_learning(image_rgb, bbox):
    """
    使用深度学习方法分析红绿灯颜色
    
    Args:
        image_rgb: 完整的 RGB 图像 (numpy array)
        bbox: 边界框 [x1, y1, x2, y2]
        
    Returns:
        tuple: (颜色名称字符串, RGB颜色元组)
    """
    # 如果 PyTorch 不可用，回退到 HSV 方法
    if not TORCH_AVAILABLE:
        return get_traffic_light_color_hsv(image_rgb, bbox)
    
    global _tl_color_model, _transform
    # 初始化模型（如果尚未初始化）
    if _tl_color_model is None or _transform is None:
        _tl_color_model, _transform = _init_deep_learning_model()
    
    x1, y1, x2, y2 = bbox
    
    # 裁剪 ROI (Region of Interest)
    roi_rgb = image_rgb[y1:y2, x1:x2]

    # 检查有效性
    if roi_rgb.shape[0] == 0 or roi_rgb.shape[1] == 0:
        return "UNKNOWN", COLOR_MAP["UNKNOWN"]
    
    try:
        # 预处理图像
        input_tensor = _transform(roi_rgb)
        input_tensor = input_tensor.unsqueeze(0)  # 添加批次维度
        
        # 设置模型为评估模式
        _tl_color_model.eval()
        
        # 使用 no_grad 减少内存使用
        with torch.no_grad():
            outputs = _tl_color_model(input_tensor)
            # 获取预测类别
            _, predicted = torch.max(outputs, 1)
            color_idx = predicted.item()
        
        color_str = TL_CLASSES[color_idx]
        return color_str, COLOR_MAP[color_str]
    except Exception as e:
        print(f"深度学习分类出错: {e}，回退到HSV方法")
        # 出错时回退到 HSV 方法
        return get_traffic_light_color_hsv(image_rgb, bbox)

def get_traffic_light_color_hsv(image_rgb, bbox):
    """
    使用传统HSV颜色空间方法分析红绿灯颜色
    
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

# 默认接口 - 使用深度学习方法
def get_traffic_light_color(image_rgb, bbox, method='deep_learning'):
    """
    分析图像区域内的颜色，判断红绿灯状态。
    
    Args:
        image_rgb: 完整的 RGB 图像 (numpy array)
        bbox: 边界框 [x1, y1, x2, y2]
        method: 识别方法，可选 'deep_learning' 或 'hsv'
        
    Returns:
        tuple: (颜色名称字符串, RGB颜色元组)
    """
    if method == 'deep_learning':
        return get_traffic_light_color_deep_learning(image_rgb, bbox)
    else:
        return get_traffic_light_color_hsv(image_rgb, bbox)

# === YOLO直接分类方法支持 ===
def register_yolo_tl_classification(results, model_names):
    """
    处理 YOLO 直接返回的红绿灯分类结果（如果模型支持）
    
    Args:
        results: YOLO 检测结果
        model_names: YOLO 模型的类别名称字典
        
    Returns:
        dict: 包含检测到的红绿灯及其状态的字典
    """
    tl_results = {}
    
    # 检查是否有红绿灯相关的细分类别
    tl_subclasses = {}
    for idx, name in model_names.items():
        if 'traffic light' in name.lower():
            # 检查是否有颜色信息
            if any(color.lower() in name.lower() for color in ['red', 'yellow', 'green']):
                for color in ['red', 'yellow', 'green']:
                    if color in name.lower():
                        tl_subclasses[idx] = color.upper()
                        break
    
    # 处理 YOLO 结果
    for r in results:
        boxes = r.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            # 检查是否是红绿灯子类
            if class_id in tl_subclasses:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                color = tl_subclasses[class_id]
                tl_results[tuple(xyxy)] = {'color': color, 'conf': conf}
            # 或者是普通红绿灯类别
            elif model_names.get(class_id) == YOLO_TARGET_CLASS:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                # 这种情况需要后续再用其他方法判断颜色
                tl_results[tuple(xyxy)] = {'color': 'UNKNOWN', 'conf': conf}
    
    return tl_results

# === 红绿灯检测结果输出接口 ===
def output_traffic_light_results(tl_results, image_rgb=None, output_format='print', output_path=None):
    """
    输出红绿灯检测结果的接口
    
    Args:
        tl_results: 红绿灯检测结果字典，格式为 {bbox_tuple: {'color': color_str, 'conf': confidence}}
        image_rgb: (可选) RGB图像，用于可视化
        output_format: 输出格式，可选 'print' (打印到控制台)、'json' (返回JSON对象)、'image' (可视化到图像)
        output_path: (可选) 当output_format为'image'时，保存图像的路径
        
    Returns:
        根据output_format返回不同内容：
        - 'print': None (直接打印到控制台)
        - 'json': dict (JSON格式的检测结果)
        - 'image': numpy array (可视化后的图像)
    """
    if not tl_results:
        if output_format == 'print':
            print("未检测到红绿灯")
        elif output_format == 'json':
            return {"detections": [], "count": 0}
        elif output_format == 'image' and image_rgb is not None:
            return image_rgb.copy()
        return None
    
    # 准备JSON格式结果
    json_results = {
        "detections": [],
        "count": len(tl_results)
    }
    
    for bbox, info in tl_results.items():
        detection = {
            "bbox": list(bbox),
            "color": info['color'],
            "confidence": info['conf']
        }
        json_results["detections"].append(detection)
    
    # 根据输出格式处理结果
    if output_format == 'print':
        print(f"检测到 {len(tl_results)} 个红绿灯:")
        for i, (bbox, info) in enumerate(tl_results.items(), 1):
            x1, y1, x2, y2 = bbox
            print(f"  {i}. 位置: [{x1}, {y1}, {x2}, {y2}], 颜色: {info['color']}, 置信度: {info['conf']:.2f}")
        return None
    
    elif output_format == 'json':
        return json_results
    
    elif output_format == 'image' and image_rgb is not None:
        # 创建图像副本进行可视化
        result_image = image_rgb.copy()
        
        for bbox, info in tl_results.items():
            x1, y1, x2, y2 = bbox
            color = COLOR_MAP.get(info['color'], COLOR_MAP["UNKNOWN"])
            
            # 绘制边界框
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            label = f"{info['color']}: {info['conf']:.2f}"
            cv2.putText(result_image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 保存图像（如果指定了路径）
        if output_path:
            # 转换为BGR格式保存
            result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, result_image_bgr)
            print(f"结果图像已保存至: {output_path}")
        
        return result_image
    
    return None

# === 综合红绿灯检测接口 ===
def detect_traffic_lights(image_rgb, yolo_results=None, model_names=None, method='deep_learning'):
    """
    综合红绿灯检测接口，整合YOLO检测和颜色分析
    
    Args:
        image_rgb: RGB图像
        yolo_results: (可选) YOLO检测结果
        model_names: (可选) YOLO模型类别名称字典
        method: 颜色分析方法，可选 'deep_learning' 或 'hsv'
        
    Returns:
        dict: 红绿灯检测结果字典
    """
    tl_results = {}
    
    # 如果提供了YOLO结果，优先处理
    if yolo_results is not None and model_names is not None:
        tl_results = register_yolo_tl_classification(yolo_results, model_names)
        
        # 对颜色为UNKNOWN的红绿灯使用指定方法进行颜色分析
        for bbox, info in tl_results.items():
            if info['color'] == 'UNKNOWN':
                color_str, _ = get_traffic_light_color(image_rgb, list(bbox), method)
                tl_results[bbox]['color'] = color_str
    
    return tl_results