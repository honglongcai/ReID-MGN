import cv2
import os
import numpy as np

def process_image(image, gamma=1.3, scale_brightness=1.25):
    # change brightness
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2] * scale_brightness
    v[v > 255] = 255
    v[v < 0] = 0
    hsv[:, :, 2] = v
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.uint8)

    # gamma correction
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 \
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

image_path = os.path.abspath('/Users/honglongcai/Desktop/f.jpg')
to_path = os.path.abspath('/Users/honglongcai/Desktop/f_gamma.jpg')
img = cv2.imread(image_path)
save_img = process_image(img, gamma=2.0)
cv2.imwrite(to_path, save_img)