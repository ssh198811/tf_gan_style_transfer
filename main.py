#coding: utf-8
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
from model_configuration_file.model_path import model_path_dict
import sys
import cv2

if __name__ == "__main__":

    import argparse

    # parser = argparse.ArgumentParser(description='generator image.')
    # parser.add_argument('--min_factor', required=False,type=int, default=16, help='min scale factor')
    # parser.add_argument('--img_path', type=str, default='./source/', help='Which picture do you want to work on')
    # parser.add_argument('--use_style', type=str, default='as', help='Which style you want to transfer')
    # parser.add_argument('--save_dir', required=False, type=str, default='./predict_img/', help='which dir you want to save img')
    # parser.add_argument('--save_name', required=False,type=str, default=None, help='save img name')
    # args = parser.parse_args()
    ###########参数合法性处理############
    sys_default_args = [16, "./source/", "as", 100, "./predict_img/", None]
    min_factor = sys_default_args[0]
    img_path = sys_default_args[1]
    use_style = sys_default_args[2]
    lerp_factor = sys_default_args[3]
    save_dir = sys_default_args[4]
    save_name = sys_default_args[5]

    args_len = len(sys.argv)

    if args_len > 1:
        if sys.argv[1] != None:
            min_factor = int(sys.argv[1])
    if args_len > 2:
        if sys.argv[2] != None:
            img_path = sys.argv[2]
    if args_len > 3:
        if sys.argv[3] != None:
            use_style = sys.argv[3]
    if args_len > 4:
        if sys.argv[4] != None:
            lerp_factor = sys.argv[4]
    if args_len > 5:
        if sys.argv[5] != None:
            save_dir = sys.argv[5]
    if args_len > 6:
        if sys.argv[6] != None:
            save_name = sys.argv[6]

    if min_factor%16!=0:
        min_factor=16
    img_path_list = []
    if os.path.exists(img_path):
        if os.path.isfile(img_path):
            _,img_name=os.path.split(img_path)
            img_path_list.append(img_path)
        elif os.path.isdir(img_path):
            img_name=[]
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
    model_names_list=list(model_path_dict.keys())
    print(model_names_list)
    if use_style not in model_names_list:
        use_style='miyaziki'
    parameter_dict = np.load(model_path_dict[use_style]).item()
    model = Generator(in_chanel=3, out_chanel=3, parameter_dict=parameter_dict)

    if len(img_path_list)>1:
        save_name=None
    if isinstance(img_name,str):
        img_name=[img_name]
    for i,path in enumerate(img_path_list):
        test_path =img_path_list[i]
        test_img = np.array(io.imread(test_path), dtype=np.float32) / 255.0 * 2.0 - 1
        h, w, c = np.shape(test_img)
        print(h,w)
        if c==1:
            test_img=np.concatenate([test_img,test_img,test_img],axis=2)
        elif c>3:
             test_img = test_img[:, :, :3]
             c=3
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
        pre_img = pre_img[h_pad_up:h_pad_up + h, w_pad_left:w_pad_left + w,:c]
        #############################存图片#############################################################
        if save_name==None:
            save_path = save_dir+img_name[i]
        else:
            save_path = save_dir+save_name
        save_sample(pre_img, save_path)

        # 进行lerp操作
        lerp_factor = float(lerp_factor)/100.0
        alpha = 1.0 - lerp_factor
        beta = lerp_factor
        gamma = 0
        img_src = cv2.imread(test_path)
        img_dst = cv2.imread(save_path)
        img_add = cv2.addWeighted(img_src, alpha, img_dst, beta, gamma)
        cv2.imwrite(save_path, img_add)
        file_dir, file_name = os.path.split(save_path)
        print("%s is save on %s dir"%(file_name,file_dir))