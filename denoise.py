import cv2
import numpy as np


def salt_and_pepper(img):
    # Remove salt & pepper noise
    img = cv2.medianBlur(img, 3)
    return img



def remove_gaussian_noise(img, kernel_size=(3, 3), sigma=1.0, h=4, brightness_thresh=220):
    # extract the luminance channel
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    white_mask = cv2.threshold(l_channel, brightness_thresh, 255, cv2.THRESH_BINARY)[1]

    #refine the mask (remove small noise, fill gaps)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

    # Apply Non-Local Means (NLM)
    img_nlm = cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)

    # Apply Gaussian blur for the rest of the image
    img_gaussian = cv2.GaussianBlur(img, kernel_size, sigma)

    # Combine results
    denoised_img = np.where(white_mask[..., None] == 255, img_nlm, img_gaussian)

    return denoised_img.astype(np.uint8)

