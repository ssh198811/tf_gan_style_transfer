#coding: utf-8
'''
Author: Naive Wu
Time: Feb 22, 2019
Target: Merge Distance
'''

from skimage.io import imsave
import numpy as np


def save_sample(np_img, img_path):
    np_img=np.round((np_img+1.0)/2.0*255)
    np_img=np.clip(np_img,0,255)
    np_img=np_img.astype(np.uint8)
    imsave(img_path, np_img)
