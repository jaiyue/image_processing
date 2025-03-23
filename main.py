import cv2
import os
import argparse
from perspective import perspective
from denoise import salt_and_pepper,remove_gaussian_noise, post_denoise, remove_gaussian_noise
from bright import adjust_contrast, white_balance, adjust_brightness, rainy_day_enhance, snowy_day_enhance, add_grain, sharpen_image, enhance_sharpness # 新增导入
from repair import repair_image


def parse_arguments():
    parser = argparse.ArgumentParser(description='Image Processing')
    parser.add_argument('input_path', type=str, help='输入目录路径')
    return parser.parse_args()

def load_images(input_path):
    valid_exts = ('.png', '.jpg', '.jpeg')
    if os.path.isfile(input_path):
        img = cv2.imread(input_path)
        if img is not None and input_path.lower().endswith(valid_exts):
            return [(img, os.path.basename(input_path))]
        else:
            return []
    
    # 如果是目录
    images = []
    for f in os.listdir(input_path):
        path = os.path.join(input_path, f)
        if os.path.isfile(path) and f.lower().endswith(valid_exts):
            img = cv2.imread(path)
            if img is not None:
                images.append((img, f))
    return images

def processing_pipeline(img):
    """处理流水线：去噪 + 校正 + 颜色均衡 + 细节增强"""
    adjust_type = "未调整"
    # 1️⃣ **初步去噪**（去除椒盐噪声）

    img = salt_and_pepper(img)
    img = remove_gaussian_noise(img)

    # 2️⃣ **几何校正**（透视变换，修正形状）
    img = perspective(img)
    
    img = repair_image(img)
    #img = adjust_contrast(img)
    
    #img = enhance_sharpness(img)
    # 4️⃣ **对比度 & 亮度增强**
    img = adjust_contrast(img)
    img = white_balance(img)
    img = adjust_brightness(img)
    img = add_grain(img)

    # 3️⃣ **颜色通道均衡**（调整亮度 & 色彩平衡）

    # 6️⃣ **精细去噪**（去除高斯噪声 + 频域降噪）
    #img = remove_gaussian_noise(img)

    # 7️⃣ **锐化处理**（恢复细节 & 提高清晰度）
    #img = post_denoise(img)

    return img, adjust_type

def save_results(processed_data, output_dir="Results"):
    os.makedirs(output_dir, exist_ok=True)
    for img, filename, adjust_type in processed_data:
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}_corrected{ext}")
        cv2.imwrite(output_path, img)
        print(f"✅ 已保存: {output_path}")
        print(f"🔧 处理方案: {adjust_type}")
        print("─" * 40)

if __name__ == "__main__":
    args = parse_arguments()
    image_data = load_images(args.input_path)
    
    processed = []
    for img, filename in image_data:
        try:
            processed_img, adjust_type = processing_pipeline(img)
            processed.append((processed_img, filename, adjust_type))
        except Exception as e:
            print(f"❌ 处理失败 {filename}: {str(e)}")
    
    if processed:
        save_results(processed)
        print(f"🎉 已完成 {len(processed)} 张图像处理")
    else:
        print("⚠️ 没有成功处理的图像")