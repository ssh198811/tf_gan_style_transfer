from utils import text_filter
import cv2
import numpy as np


def read_img(img_path):
    if text_filter.is_contain_chinese(img_path) is True:
        return cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
    else:
        return cv2.imread(img_path)

def lerp_img(src_path, dst_path, lerp_value):
    # 进行lerp操作
    alpha = 1.0 - float(lerp_value) / 100.0
    beta = float(lerp_value) / 100.0
    gamma = 0
    img_src = read_img(src_path)
    img_dst = read_img(dst_path)
    img_add = cv2.addWeighted(img_src, alpha, img_dst, beta, gamma)
    return img_add