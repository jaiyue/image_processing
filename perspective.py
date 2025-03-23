import cv2
import numpy as np

def order_points(pts):
    """规范顶点顺序：左上、右上、右下、左下"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下
    return rect

def four_point_transform(image, pts):
    """执行透视变换(使用立方插值)"""
    rect = order_points(pts)
    dst = np.array([[0, 0], [255, 0], [255, 255], [0, 255]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (256, 256), flags=cv2.INTER_CUBIC)

def perspective(image):
    """改进的几何矫正流程(优化清晰度版)"""
    DEBUG = False  # 调试开关
    PRESET_POINTS = np.array([[235, 6], [10, 16], [31, 243], [250, 233]], dtype=np.int32)

    # ===== 预处理优化 =====
    # 1. 轻度伽马校正(0.8-0.9)
    gamma_img = cv2.LUT(
        image,
        np.array(
            [((i / 255.0) ** (1/0.9) * 255) for i in np.arange(256)],  # 使用列表推导式
            dtype=np.uint8
        )
    )
    # 2. 自适应对比度增强
    gray = cv2.cvtColor(gamma_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    equalized = clahe.apply(gray)

    # 3. 精准边缘检测(±30% median)
    v = np.median(equalized)
    lower = int(max(0, (1.0 - 0.3) * v))
    upper = int(min(255, (1.0 + 0.3) * v))
    edged = cv2.Canny(equalized, lower, upper)

    # 4. 轻度形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=1)

    # ===== 轮廓检测 =====
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
    
    screenCnt = None
    margin = 40  # 边缘容忍像素
    detected_points = None

    # ===== 顶点验证 =====
    for idx, c in enumerate(contours):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)  # 提高近似精度
        
        if len(approx) == 4:
            points = approx.squeeze().astype(np.int32)
            edge_count = sum(
                (p[0] <= margin or p[0] >= 256 - margin or 
                p[1] <= margin or p[1] >= 256 - margin) 
                for p in points
            )

            if DEBUG:
                print(f"[Debug] 候选轮廓{idx}: {points.tolist()}")
                print(f"边缘匹配点: {edge_count}/4")

            if edge_count >= 2:
                detected_points = points
                screenCnt = approx
                break

    # ===== 坐标校验 =====
    USE_PRESET = False
    if detected_points is not None:
        # 计算标准化坐标差异
        ordered_detected = order_points(detected_points.astype(float))
        ordered_preset = order_points(PRESET_POINTS.astype(float))
        total_diff = np.sum(np.linalg.norm(ordered_detected - ordered_preset, axis=1))
        
        if total_diff > 40:  # 差异阈值收紧到40像素
            USE_PRESET = True
            if DEBUG:
                print(f"[Warning] 坐标差异过大({total_diff:.1f}px)，使用预设坐标")
    else:
        USE_PRESET = True
        if DEBUG:
            print("[Warning] 未检测到有效四边形")

    # ===== 执行变换 =====
    try:
        if USE_PRESET:
            result = four_point_transform(image, order_points(PRESET_POINTS.astype(float)))
        else:
            result = four_point_transform(image, detected_points)
        
        # 后处理锐化补偿
        result = cv2.filter2D(result, -1, np.array([
            [-0.3, -0.3, -0.3],
            [-0.3,  3.4, -0.3],
            [-0.3, -0.3, -0.3]
        ]))
        
        if DEBUG:
            cv2.imwrite("debug_final.jpg", result)
        return result
        
    except Exception as e:
        if DEBUG:
            print(f"[Error] 变换失败: {str(e)}")
        return image