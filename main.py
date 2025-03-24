import cv2
import os
import argparse
from perspective import perspective
from denoise import salt_and_pepper,remove_gaussian_noise
from bright import adjust_contrast, white_balance, adjust_brightness, add_grain, sharpen_image, highlight_blue_boost, adjust_saturation,boost_red_green_in_shadows # 新增导入
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
    #初步去噪
    img = salt_and_pepper(img)
    
    img = remove_gaussian_noise(img)

    #几何校正
    img = perspective(img)
    
    img = repair_image(img)
    img = sharpen_image(img)

    #  对比度，亮度
    img = adjust_contrast(img)
    img = white_balance(img)
    img = adjust_saturation(img)
    img = adjust_brightness(img)
    img = highlight_blue_boost(img)
    img = boost_red_green_in_shadows(img)
    img = add_grain(img)

    return img

def save_results(processed_data, output_dir="Results"):
    os.makedirs(output_dir, exist_ok=True)
    for img, filename in processed_data:
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}_corrected{ext}")
        cv2.imwrite(output_path, img)
        print(f"✅ 已保存: {output_path}")
        print("─" * 40)

if __name__ == "__main__":
    args = parse_arguments()
    image_data = load_images(args.input_path)
    
    processed = []
    for img, filename in image_data:
        try:
            processed_img = processing_pipeline(img)
            processed.append((processed_img, filename))
        except Exception as e:
            print(f"❌ 处理失败 {filename}: {str(e)}")
    
    if processed:
        save_results(processed)
        print(f"🎉 已完成 {len(processed)} 张图像处理")
    else:
        print("⚠️ 没有成功处理的图像")