import cv2
import numpy as np

def salt_and_pepper(img):
    # Remove salt & pepper noise
    img = cv2.medianBlur(img, 3)
    return img
  
def remove_gaussian_noise(img, kernel_size=(5, 5), sigma=1.0):
    # Remove gaussian noise using blur
    denoised_img = cv2.GaussianBlur(img, kernel_size, sigma)
    return denoised_img