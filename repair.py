import cv2
import numpy as np

def detect_black_defect(img):
    """精确检测右上角黑色缺失区域"""
    h, w = img.shape[:2]
    
    # 限定检测区域（右半图的顶部25%）
    roi = img[0:int(h*0.25), w//2:w]
    
    # 转换到HSV空间检测纯黑区域
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 10])
    mask = cv2.inRange(hsv, lower_black, upper_black)
    
    # 精细化形态学处理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 精确轮廓分析
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(max_contour, True)
        area = cv2.contourArea(max_contour)
        if perimeter == 0:
            return None
        circularity = 4 * np.pi * area / (perimeter**2)
        
        if circularity > 0.7:
            (x, y), r = cv2.minEnclosingCircle(max_contour)
            x += w//2  # 调整坐标为整个图像的坐标系
            return (int(x), int(y), int(r * 1.2))  # 增加半径以确保覆盖整个缺失区域
    return None

def inpaint_image(img, defect_params):
    """使用简单的插值方法修复缺失区域"""
    if defect_params is None:
        return img  # 如果没有缺失区域，直接返回原图

    # 获取缺失区域的圆形参数 (x, y, r)
    x, y, r = defect_params
    
    # 创建掩膜，标记缺失区域
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.circle(mask, (x, y), r, (255, 255, 255), -1)  # 在掩膜上绘制圆形
    
    # 使用inpaint方法填补缺失区域
    repaired_img = cv2.inpaint(img, mask[:,:,0], 3, cv2.INPAINT_TELEA)
    return repaired_img

def repair_image(img):
    defect_params = detect_black_defect(img)
    if defect_params:
        x, y, r = defect_params
        mask = np.zeros_like(img, dtype=np.uint8)
        cv2.circle(mask, (x, y), r, (255, 255, 255), -1)
        repaired_img = cv2.inpaint(img, mask[:,:,0], 5, cv2.INPAINT_NS)
        return repaired_img
    else:
        print("没有检测到缺陷")  # 未检测到缺陷时提示
    return img