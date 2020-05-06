# coding: utf-8
'''
Author: Naive Wu
Time: APR 16, 2019
Target: 生成风格图片
'''
import os
from skimage import io
import numpy as np
import tensorflow as tf
from model.Generator import Generator
from utils.save_img import save_sample
# from model_configuration_file.model_path import model_path_dict
import sys
import cv2
import loadDict
from utils import img_process
import time

def init_tf_config():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

def process_img(params=[16, "./source/", "as", 100, "./predict_img/", None]):
    args_len = len(params)

    if args_len > 0:
        if params[0] is not None:
            min_factor = int(params[0])
    if args_len > 1:
        if params[1] is not None:
            img_path = params[1]
    if args_len > 2:
        if params[2] is not None:
            use_style = params[2]
    if args_len > 3:
        if params[3] is not None:
            lerp_factor = params[3]
    if args_len > 4:
        if params[4] is not None:
            save_dir = params[4]
    if args_len > 5:
        if params[5] is not None:
            save_name = params[5]

    if min_factor % 16 != 0:
        min_factor = 16
    img_path_list = []
    if os.path.exists(img_path):
        if os.path.isfile(img_path):
            _, img_name = os.path.split(img_path)
            img_path_list.append(img_path)
        elif os.path.isdir(img_path):
            img_name = []
            for file in os.listdir(img_path):
                file_path = os.path.join(img_path, file)
                if os.path.isfile(file_path):
                    img_name.append(file)
                    img_path_list.append(file_path)
        else:
            raise TypeError("img_path must be file or dir")
    else:
        raise FileExistsError("img_path do not exist")
    ###########################载入模型###############################################################
    model_path_dict = {}
    for pair in loadDict.load_dict():
        key = pair['name']
        value = pair['model']
        model_path_dict[key] = value

    model_names_list = list(model_path_dict.keys())
    print(model_names_list)
    if use_style not in model_names_list:
        use_style = 'miyaziki'
    parameter_dict = np.load(model_path_dict[use_style]).item()
    model = Generator(in_chanel=3, out_chanel=3, parameter_dict=parameter_dict)

    if len(img_path_list) > 1:
        save_name = None
    if isinstance(img_name, str):
        img_name = [img_name]
    for i, path in enumerate(img_path_list):
        test_path = img_path_list[i]
        print("%s is under processing" % test_path)
        time_start = time.time()
        test_img = np.array(io.imread(test_path), dtype=np.float32) / 255.0 * 2.0 - 1
        h, w, c = np.shape(test_img)
        print(h, w)
        if c == 1:
            test_img = np.concatenate([test_img, test_img, test_img], axis=2)
        elif c > 3:
            test_img = test_img[:, :, :3]
            c = 3
            print('chanel num larger 3,not rgb')
        ###########填补操作####################
        h_pad = np.ceil(h / min_factor) * min_factor - h
        if h_pad < 6 and h_pad != 0:
            h_pad += min_factor
        w_pad = np.ceil(w / min_factor) * min_factor - w
        if w_pad < 6 and w_pad != 0:
            w_pad += min_factor
        h_pad_up = int(h_pad // 2)
        h_pad_down = int(h_pad - h_pad_up)
        w_pad_left = int(w_pad // 2)
        w_pad_right = int(w_pad - w_pad_left)
        test_img = np.reshape(test_img, [1, h, w, 3])
        if h_pad == 0 and w_pad >= 6:
            test_img = tf.pad(test_img, [[0, 0], [0, 0], [3, 3], [0, 0]], mode="REFLECT")
            test_img = tf.pad(test_img, [[0, 0], [0, 0], [w_pad_left - 3, w_pad_right - 3], [0, 0]])
        elif w_pad == 0 and h_pad >= 6:
            test_img = tf.pad(test_img, [[0, 0], [3, 3], [0, 0], [0, 0]], mode="REFLECT")
            test_img = tf.pad(test_img, [[0, 0], [h_pad_up - 3, h_pad_down - 3], [0, 0], [0, 0]])
        elif h_pad >= 6 and w_pad >= 6:
            test_img = tf.pad(test_img, [[0, 0], [3, 3], [3, 3], [0, 0]], mode="REFLECT")
            test_img = tf.pad(test_img,
                              [[0, 0], [h_pad_up - 3, h_pad_down - 3], [w_pad_left - 3, w_pad_right - 3], [0, 0]])
        pre_img = model(test_img)
        pre_img = pre_img[0]
        pre_img = pre_img[h_pad_up:h_pad_up + h, w_pad_left:w_pad_left + w, :c]

        time_end = time.time()
        print('time cost', time_end - time_start, 's')
        #############################存图片#############################################################
        if save_name is None:
            save_path = save_dir + img_name[i]
        else:
            save_path = save_dir + save_name
        save_sample(pre_img, save_path)

        # 进行lerp操作
        img_add, ret = img_process.lerp_img(test_path, save_path, lerp_factor)
        if ret == 0:
            cv2.imwrite(save_path, img_add)
            file_dir, file_name = os.path.split(save_path)
            print("%s is save on %s dir" % (file_name, file_dir))
        else:
            print("generate failed when processing %s" % test_path)
