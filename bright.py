import cv2
import numpy as np

# ------------------ 基础调整函数 ------------------
def white_balance(img, blue_boost=0.1, green_boost=0.05):
    """白平衡调整（LAB颜色空间）"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 全局蓝色/绿色增强
    a = cv2.addWeighted(a, 1.0, np.full_like(a, 10), green_boost, 0)
    b = cv2.addWeighted(b, 1.0, np.full_like(b, 10), blue_boost, 0)
    
    # 通道平滑处理
    a = cv2.GaussianBlur(a, (3,3), 0)
    b = cv2.GaussianBlur(b, (3,3), 0)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def add_grain(img, intensity=0.06):
    """增加胶片颗粒效果"""
    noise = np.random.normal(0, intensity*255, img.shape)
    noisy_img = np.clip(img.astype(np.float32) + noise, 0, 255)
    return noisy_img.astype(np.uint8)

def adjust_contrast(img, clip_limit=1):
    """对比度调整（CLAHE）"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(12,12))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)

def adjust_brightness(img, gamma=1.0):
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
    
    # 仅处理亮度通道
    inv_gamma = 1.0 / max(gamma, 0.1)  # 防止除以零
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    l_adjusted = cv2.LUT(l_channel, table)
    
    # 合并通道并转换回BGR
    adjusted_lab = cv2.merge([l_adjusted, a_channel, b_channel])
    result = cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)
    
    # 防止过曝的最终裁剪
    return np.clip(result, 0, 255).astype(np.uint8)

def sharpen_image(img, blur_kernel=(5,5), sigma=0, alpha=1.5, beta=-0.5):
    """改进的锐化函数（避免颜色偏移）"""
    blurred = cv2.GaussianBlur(img, blur_kernel, sigmaX=sigma)
    sharpened = cv2.addWeighted(img, alpha, blurred, beta, 0)
  
    return sharpened