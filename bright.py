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

# ------------------ 场景处理函数 ------------------
def rainy_day_enhance(img):
    """雨天场景增强（增加绿色通道）"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 增强绿色通道
    a = cv2.addWeighted(a, 1.0, np.full_like(a, 15), 0.2, 0)
    # 降低亮度
    l = cv2.addWeighted(l, 0.9, np.zeros_like(l), 0, 0)
    
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def snowy_day_enhance(img):
    """雪天场景增强（增强蓝色通道）"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 增强蓝色通道
    b = cv2.addWeighted(b, 1.0, np.full_like(b, 25), 0.5, 0)
    # 提升亮度
    l = cv2.addWeighted(l, 1.1, np.zeros_like(l), 0, 0)
    
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

# ------------------ 智能处理管道 ------------------
def processing_pipeline(img, scene_type="normal"):
    """完整处理流程示例"""
    # 基础调整
    img = adjust_contrast(img)[0]
    img = white_balance(img)[0]
    img = adjust_brightness(img, 1.05)
    
    # 场景增强
    if scene_type == "rainy":
        img = rainy_day_enhance(img)[0]
    elif scene_type == "snowy":
        img = snowy_day_enhance(img)[0]
        
    # 可选添加颗粒
    # img = add_grain(img, 0.03)
    
    return img
        
def sharpen_image(img, blur_kernel=(5,5), sigma=0, alpha=1.5, beta=-0.5):
    """改进的锐化函数（避免颜色偏移）"""
    # 方法一：使用非锐化掩模（推荐）
    blurred = cv2.GaussianBlur(img, blur_kernel, sigmaX=sigma)
    sharpened = cv2.addWeighted(img, alpha, blurred, beta, 0)
    
    # 方法二：LAB颜色空间仅锐化亮度通道（可选）
    # lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # l_channel, a_channel, b_channel = cv2.split(lab)
    # blurred_l = cv2.GaussianBlur(l_channel, blur_kernel, sigmaX=sigma)
    # sharpened_l = cv2.addWeighted(l_channel, alpha, blurred_l, beta, 0)
    # sharpened = cv2.cvtColor(cv2.merge([sharpened_l, a_channel, b_channel]), cv2.COLOR_LAB2BGR)
    
    return sharpened

def enhance_sharpness(img, intensity=0.8, radius=1.0, threshold=5):
    """
    改进的清晰度增强函数（LAB颜色空间处理）
    
    参数说明：
    - intensity: 锐化强度 (0.5-2.0)
      0.5: 轻微锐化 | 1.0: 标准锐化 | 2.0: 强烈锐化
    - radius: 锐化作用半径 (0.5-2.5)
      控制锐化影响的边缘宽度（值越大影响区域越宽）
    - threshold: 锐化阈值 (0-15)
      抑制平坦区域的伪影（值越大锐化越保守）
    """
    # 转换到LAB颜色空间处理亮度通道
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    
    # 生成高斯模糊层（控制锐化作用半径）
    blurred = cv2.GaussianBlur(l_channel, (0, 0), radius)
    
    # 计算锐化蒙版（高频细节提取）
    sharp_mask = cv2.subtract(l_channel, blurred)
    
    # 应用自适应阈值（抑制平坦区域噪点）
    _, sharp_mask = cv2.threshold(
        sharp_mask, 
        threshold, 
        0, 
        cv2.THRESH_TOZERO
    )
    
    # 增强亮度通道
    enhanced_l = cv2.addWeighted(
        l_channel, 
        1.0, 
        sharp_mask, 
        intensity, 
        0
    )
    
    # 合并通道并转换回BGR
    enhanced_lab = cv2.merge([enhanced_l, a, b])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)