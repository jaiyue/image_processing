import cv2
import numpy as np

def salt_and_pepper(img):
    """去除椒盐 & 高斯噪声"""
    img = cv2.medianBlur(img, 3)  # 先去椒盐噪声
    return img
  
def remove_gaussian_noise(img, h_base=3, h_white=5, brightness_thresh=225):
    """
    改进版非局部均值去噪，对白色区域增强处理
    参数：
        h_base: 常规区域去噪强度（默认3）
        h_white: 白色区域去噪强度（默认8）
        brightness_thresh: 白色区域亮度阈值（0-255，默认220）
    """
    # 转换到LAB颜色空间获取亮度通道
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:,:,0]

    # 创建白色区域掩膜（亮度高于阈值）
    white_mask = cv2.threshold(l_channel, brightness_thresh, 255, cv2.THRESH_BINARY)[1]
    
    # 形态学处理优化掩膜（消除小孔洞和毛刺）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    
    # 分区域处理
    denoised_base = cv2.fastNlMeansDenoisingColored(
        img, None, 
        h=h_base, hColor=h_base,
        templateWindowSize=7, 
        searchWindowSize=21
    )
    
    # 仅对白色区域应用强力去噪
    denoised_strong = cv2.fastNlMeansDenoisingColored(
        img, None,
        h=h_white, hColor=h_white//2,
        templateWindowSize=9,
        searchWindowSize=35
    )
    
    # 混合结果（白色区域使用强去噪，其他区域使用常规去噪）
    final_img = np.where(
        white_mask[..., None].astype(bool), 
        denoised_strong, 
        denoised_base
    ).astype(np.uint8)
    
    return final_img