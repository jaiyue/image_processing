import cv2
import numpy as np

def white_balance(img):
    #White balance adjustment
    result = img.copy()
    
    avg_b = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_r = np.mean(img[:, :, 2])

    avg_gray = (avg_b + avg_g + avg_r) / 3
    gain_b = avg_gray / avg_b
    gain_g = avg_gray / avg_g
    gain_r = avg_gray / avg_r

    result[:, :, 0] = np.clip(img[:, :, 0] * gain_b, 0, 255)
    result[:, :, 1] = np.clip(img[:, :, 1] * gain_g, 0, 255)
    result[:, :, 2] = np.clip(img[:, :, 2] * gain_r, 0, 255)

    return result.astype(np.uint8)

def add_grain(img, intensity=0.05):
    """Add film grain effect"""
    noise = np.random.normal(0, intensity*255, img.shape)
    noisy_img = np.clip(img.astype(np.float32) + noise, 0, 255)
    return noisy_img.astype(np.uint8)

def adjust_contrast(img, clip_limit=0.4):
    #Contrast adjustment
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(9,9))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)

def highlight_blue_boost(img, threshold=235, blue_boost=20):
    #Enhance blue in highlight areas
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    highlight_mask = (l > threshold).astype(np.uint8)
    b = cv2.subtract(b, blue_boost * highlight_mask)

    adjusted_lab = cv2.merge([l, a, b])
    return cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)

def darken_highlights(img, threshold=235, darken_amount=20):
    # Darken highlight areas
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Create a mask for bright areas
    highlight_mask = (l > threshold).astype(np.uint8)
    
    # Reduce brightness in highlight areas
    l = cv2.subtract(l, darken_amount * highlight_mask)

    # Merge adjusted channels and convert back to BGR
    adjusted_lab = cv2.merge([l, a, b])
    return cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)


def adjust_brightness(img, gamma=1.3):
    """Adjust brightness in LAB color space
    gamma < 1.0: brighten image
    gamma = 1.0: no change
    gamma > 1.0: darken image
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    median_l = np.median(l_channel)
    gamma = gamma * (128.0 / max(median_l, 1))

    inv_gamma = 1.0 / max(gamma, 0.1)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    l_adjusted = cv2.LUT(l_channel, table)

    adjusted_lab = cv2.merge([l_adjusted, a_channel, b_channel])
    result = cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2BGR)
    
    return np.clip(result, 0, 255).astype(np.uint8)

def sharpen_image(img, blur_kernel=(3,3), sigma=0, alpha=1.6, beta=-0.6):
    """Improved sharpening function that avoids color shifts"""
    blurred = cv2.GaussianBlur(img, blur_kernel, sigmaX=sigma)
    sharpened = cv2.addWeighted(img, alpha, blurred, beta, 0)
  
    return sharpened