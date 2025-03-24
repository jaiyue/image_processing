import cv2
import numpy as np

# ------------------ 基础调整函数 ------------------
def white_balance(img):
    """ 基于灰度世界假设的白平衡调整 """
    result = img.copy()
    
    # 计算每个通道的均值
    avg_b = np.mean(img[:, :, 0])  # B通道均值
    avg_g = np.mean(img[:, :, 1])  # G通道均值
    avg_r = np.mean(img[:, :, 2])  # R通道均值

    # 计算白平衡增益系数
    avg_gray = (avg_b + avg_g + avg_r) / 3
    gain_b = avg_gray / avg_b
    gain_g = avg_gray / avg_g
    gain_r = avg_gray / avg_r

    # 乘以增益，限制范围防止溢出
    result[:, :, 0] = np.clip(img[:, :, 0] * gain_b, 0, 255)
    result[:, :, 1] = np.clip(img[:, :, 1] * gain_g, 0, 255)
    result[:, :, 2] = np.clip(img[:, :, 2] * gain_r, 0, 255)

    return result.astype(np.uint8)

def add_grain(img, intensity=0.03):
    """增加胶片颗粒效果"""
    noise = np.random.normal(0, intensity*255, img.shape)
    noisy_img = np.clip(img.astype(np.float32) + noise, 0, 255)
    return noisy_img.astype(np.uint8)

def adjust_saturation(img, factor = 0.9):
    """调整饱和度（HSV颜色空间处理）
    :param factor: 饱和度缩放因子（0.0-1.0）
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * factor, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

# ------------------ 修改对比度函数（降低对比度） ------------------
def adjust_contrast(img, clip_limit = 0.9):  # 默认值设为0.7以降低对比度
    """对比度调整（CLAHE）"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(12,12))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)

# ------------------ 新增高光区域蓝色增强函数 ------------------
def highlight_blue_boost(img, threshold=220, blue_boost=10):
    """识别高光区域并增加蓝色（LAB颜色空间处理）
    :param threshold: 高光亮度阈值（0-255）
    :param blue_boost: 蓝色增强强度（负值增强蓝色，正值增强黄色）
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 创建高光掩膜
    highlight_mask = (l > threshold).astype(np.uint8)

    # **减少 B 通道数值以增强蓝色**
    b = cv2.subtract(b, blue_boost * highlight_mask)  # 改为 subtract 以减少黄色

    # 合并通道并返回BGR图像
    adjusted_lab = cv2.merge([l, a, b])
    return cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)

def boost_red_green_in_shadows(img, threshold=100, boost_red=5, boost_green=30):
    """识别暗部区域并增加红色和绿色（LAB颜色空间处理）
    
    :param threshold: 暗部亮度阈值（0-255，值越低，选取的暗部区域越少）
    :param boost_strength: 颜色增强强度（正值增加红色和绿色）
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 创建暗部掩膜
    shadow_mask = (l < threshold).astype(np.uint8)
    
    # **增加 A（红色）和 B（绿色）通道数值**
    a = cv2.add(a, boost_red * shadow_mask)  # 增强红色
    b = cv2.add(b, boost_green * shadow_mask)  # 增强绿色

    # 合并通道并返回BGR图像
    adjusted_lab = cv2.merge([l, a, b])
    return cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)

def adjust_brightness(img, gamma=1.6):
    """
    基础亮度调节函数（LAB颜色空间处理）
    :param gamma: 亮度调节系数
      - gamma < 1.0: 提亮图像（如0.7）
      - gamma = 1.0: 无变化
      - gamma > 1.0: 压暗图像（如1.5）
    """
    # 转换到LAB颜色空间
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    median_l = np.median(l_channel)
    gamma = gamma * (128.0 / max(median_l, 1))  # 让中等亮度调整接近128

    # 计算自适应 gamma 映射表
    inv_gamma = 1.0 / max(gamma, 0.1)  # 避免除零
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    l_adjusted = cv2.LUT(l_channel, table)

    # 合并通道并转换回 BGR
    adjusted_lab = cv2.merge([l_adjusted, a_channel, b_channel])
    result = cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)
    
    # 防止过曝的最终裁剪
    return np.clip(result, 0, 255).astype(np.uint8)

def sharpen_image(img, blur_kernel=(5,5), sigma=0, alpha=1.5, beta=-0.5):
    """改进的锐化函数（避免颜色偏移）"""
    blurred = cv2.GaussianBlur(img, blur_kernel, sigmaX=sigma)
    sharpened = cv2.addWeighted(img, alpha, blurred, beta, 0)
  
    return sharpened