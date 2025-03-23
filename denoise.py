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
        h=h_white, hColor=h_white//2,  # 降低色度去噪强度避免变色
        templateWindowSize=9,           # 增大模板窗口
        searchWindowSize=35             # 扩大搜索范围
    )
    
    # 混合结果（白色区域使用强去噪，其他区域使用常规去噪）
    final_img = np.where(
        white_mask[..., None].astype(bool), 
        denoised_strong, 
        denoised_base
    ).astype(np.uint8)
    
    return final_img

def frequency_domain_denoise(channel):
    """改进频域去噪（保留更多细节）"""
    dft = cv2.dft(np.float32(channel), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = channel.shape
    crow, ccol = rows//2, cols//2

    # 生成更宽松的滤波器
    mask = np.ones((rows, cols), np.float32)
    mask = cv2.GaussianBlur(mask, (0,0), sigmaX=30)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 1  # 扩大通带区域
    mask_2ch = np.stack([mask, mask], axis=-1)

    fshift = dft_shift * mask_2ch
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    
    return cv2.normalize(
        cv2.magnitude(img_back[:,:,0], img_back[:,:,1]), 
        None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

def adaptive_sharpen(img, threshold=200):
    """自适应锐化（改进版）"""
    # 非锐化掩模增强细节
    blurred = cv2.GaussianBlur(img, (0,0), 3)
    sharpened = cv2.addWeighted(img, 1.8, blurred, -0.8, 0)
    
    # 仅在高对比度区域应用强锐化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # 混合输出
    return np.where(mask[...,None].astype(bool), sharpened, img).astype(np.uint8)

def post_denoise(img):
    """优化后处理流程"""
    # 1. 弱化各向异性扩散
    anisotropic = cv2.ximgproc.anisotropicDiffusion(img, alpha=0.1, K=30, niters=5)
    
    # 2. 改进的频域去噪
    lab = cv2.cvtColor(anisotropic, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_denoised = frequency_domain_denoise(l)
    
    # 3. 轻度NLM去噪
    l_final = cv2.fastNlMeansDenoising(l_denoised, None, h=5, templateWindowSize=3, searchWindowSize=7)
    
    # 4. 合并通道并锐化
    merged = cv2.merge([l_final, a, b])
    denoised_bgr = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    
    return adaptive_sharpen(denoised_bgr)