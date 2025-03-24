import cv2
import numpy as np

def salt_and_pepper(img):
    """去除椒盐 & 高斯噪声"""
    img = cv2.medianBlur(img, 3)  # 先去椒盐噪声
    return img
  
def remove_gaussian_noise(img, kernel_size=(5, 5), sigma=1.0):
    # 使用高斯模糊进行去噪
    denoised_img = cv2.GaussianBlur(img, kernel_size, sigma)
    
    return denoised_img