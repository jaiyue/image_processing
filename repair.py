import cv2
import numpy as np


def detect_black_defect(img):
    h, w = img.shape[:2]
    roi = img[0:int(h*0.25), w//2:w]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 10])
    mask = cv2.inRange(hsv, lower_black, upper_black)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(max_contour, True)
        area = cv2.contourArea(max_contour)
        if perimeter == 0:
            return None
        circularity = 4 * np.pi * area / (perimeter**2)

        if circularity > 0.7:
            (x, y), r = cv2.minEnclosingCircle(max_contour)
            x += w//2
            return (int(x), int(y), int(r*1.1))
    return None


def progressive_inpaint(img, defect_params, max_iters=5):
    """ 逐层修复缺陷区域 """
    if defect_params is None:
        return img

    x, y, r = defect_params
    mask = np.zeros_like(img[:, :, 0], dtype=np.uint8)
    cv2.circle(mask, (x, y), int(r), 255, -1)

    # 迭代修复
    for _ in range(max_iters):
        edges = cv2.Canny(mask, 50, 150)
        edges = cv2.dilate(edges, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (3, 3)))  # 解决边缘丢失
        mask = cv2.dilate(mask, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (3, 3)))  # 逐层扩展

        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)  # 更稳定的修复方式

    return img


def repair_image(img):
    defect_params = detect_black_defect(img)
    return progressive_inpaint(img, defect_params, max_iters=10)
