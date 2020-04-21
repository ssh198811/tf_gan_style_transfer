from utils import text_filter
import cv2
import os
import numpy as np


def read_img(img_path):
    if text_filter.is_contain_chinese(img_path) is True:
        return cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
    else:
        return cv2.imread(img_path)

def write_img(img, img_path=""):
    if os.path.exists(img_path) == True:
        os.unlink(img_path)

    if img is None:
        return

    if text_filter.is_contain_chinese(img_path) is True:
        if img_path.endswith('.jpg') is True:
            cv2.imencode('.jpg', img)[1].tofile(img_path)
        elif img_path.endswith('.png') is True:
            cv2.imencode('.png', img)[1].tofile(img_path)
    else:
         cv2.imwrite(img_path,img)

def lerp_img(src_path, dst_path, lerp_value):
    # 进行lerp操作
    alpha = 1.0 - float(lerp_value) / 100.0
    beta = float(lerp_value) / 100.0
    gamma = 0
    img_src = read_img(src_path)
    img_dst = read_img(dst_path)
    h_src, w_src, c_src = img_src.shape
    h_dst, w_dst, c_dst = img_dst.shape
    if h_src != h_dst or w_src != w_dst:
        return None, -1

    img_add = cv2.addWeighted(img_src, alpha, img_dst, beta, gamma)
    return img_add, 0
