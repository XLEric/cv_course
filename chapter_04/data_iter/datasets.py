#-*-coding:utf-8-*-
# date:2019-05-20
# Author: xiang li
# function:
import glob
import math
import os
import random
import shutil
from pathlib import Path
from PIL import Image
# import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# import json
# from support.alignment_aug import *

flag_debug = False
# 非形变处理
def letterbox(img_,size_=416,mean_rgb = (128,128,128)):

    shape_ = img_.shape[:2]  # shape = [height, width]
    ratio = float(size_) / max(shape_)  # ratio  = old / new
    new_shape_ = (round(shape_[1] * ratio), round(shape_[0] * ratio))
    dw_ = (size_ - new_shape_[0]) / 2  # width padding
    dh_ = (size_ - new_shape_[1]) / 2  # height padding
    top_, bottom_ = round(dh_ - 0.1), round(dh_ + 0.1)
    left_, right_ = round(dw_ - 0.1), round(dw_ + 0.1)
    # resize img
    img_a = cv2.resize(img_, new_shape_, interpolation=cv2.INTER_LINEAR)

    img_a = cv2.copyMakeBorder(img_a, top_, bottom_, left_, right_, cv2.BORDER_CONSTANT, value=mean_rgb)  # padded square
    # print('fix size : ',img_a.shape)
    return img_a
# 图像白化
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y

def select_device(force_cpu=False):
    if force_cpu:
        cuda = False
        device = torch.device('cpu')
    else:
        cuda = torch.cuda.is_available()
        device = torch.device('cuda:0' if cuda else 'cpu')

        if torch.cuda.device_count() > 1:
            device = torch.device('cuda' if cuda else 'cpu')
            print('Found %g GPUs' % torch.cuda.device_count())

    print('Using %s %s\n' % (device.type, torch.cuda.get_device_properties(0) if cuda else ''))
    return device
# 图像亮度、对比度增强
def contrast_img(img, c, b):  # 亮度就是每个像素所有通道都加上b
    rows, cols, channels = img.shape
    # 新建全零(黑色)图片数组:np.zeros(img1.shape, dtype=uint8)
    blank = np.zeros([rows, cols, channels], img.dtype)
    dst = cv2.addWeighted(img, c, blank, 1-c, b)
    return dst
# 图像旋转
def M_rotate_image(image , angle , cx , cy):
    '''
    图像旋转
    :param image:
    :param angle:
    :return: 返回旋转后的图像以及旋转矩阵
    '''
    (h , w) = image.shape[:2]
    # (cx , cy) = (int(0.5 * w) , int(0.5 * h))
    M = cv2.getRotationMatrix2D((cx , cy) , -angle , 1.0)
    cos = np.abs(M[0 , 0])
    sin = np.abs(M[0 , 1])

    # 计算新图像的bounding
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0 , 2] += int(0.5 * nW) - cx
    M[1 , 2] += int(0.5 * nH) - cy
    return cv2.warpAffine(image , M , (nW , nH)) , M

class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, img_size=(224,224), flag_agu = False,fix_res = True):
        print('img_size (height,width) : ',img_size[0],img_size[1])
        labels_ = []
        files_ = []
        for idx,doc in enumerate(sorted(os.listdir(path), key=lambda x:int(x.split('.')[0]), reverse=False)):
        # for idx,doc in enumerate(os.listdir(path)):
            print(' %s label is %s \n'%(doc,idx))

            for file in os.listdir(path+doc):
                if '.jpg' in file:
                    labels_.append(idx)
                    files_.append(path+doc + '/' + file)
            print()
        print('\n')

        self.labels = labels_
        self.files = files_
        self.img_size = img_size
        self.flag_agu = flag_agu
        self.fix_res = fix_res

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index]
        label_ = self.labels[index]

        img = cv2.imread(img_path)  # BGR

        cv_resize_model = [cv2.INTER_LINEAR,cv2.INTER_CUBIC,cv2.INTER_NEAREST,cv2.INTER_AREA]

        if self.flag_agu == True:
            if random.randint(0,5)==0:
                cx = int(img.shape[1]/2)
                cy = int(img.shape[0]/2)
                angle = random.randint(-45,45)
                offset_x = random.randint(-10,10)
                offset_y = random.randint(-10,10)
                if not(angle==0 and offset_x==0 and offset_y==0):
                    img,_  = M_rotate_image(img , angle , cx+offset_x , cy+offset_y)

        if self.flag_agu == True and random.randint(0,20)==1:
            resize_idx = random.randint(0,3)

            if self.fix_res:
                img_ = letterbox(img,size_=self.img_size[0],mean_rgb = (128,128,128))
            else:
                img_ = cv2.resize(img, (self.img_size[1],self.img_size[0]), interpolation = cv_resize_model[resize_idx])
        else:
            if self.fix_res:
                img_ = letterbox(img,size_=self.img_size[0],mean_rgb = (128,128,128))
            else:
                img_ = cv2.resize(img, (self.img_size[1],self.img_size[0]), interpolation = cv2.INTER_CUBIC)

        if self.flag_agu == True and random.randint(0,6)==0:
            img_ = cv2.flip(img_, random.randint(-1,1))# 0上下翻转 ，-1，上下+左右翻转 ，1左右翻转

        if self.flag_agu == True:
            if random.randint(0,16)==8:
                c = float(random.randint(80,120))/100.
                b = random.randint(-10,10)
                img_ = contrast_img(img_, c, b)

        if self.flag_agu == True:
            if random.randint(0,9)==1:# and (label_ == 15 or label_ == 16 or label_ == 17):
                # print('agu hue ')
                img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
                hue_x = random.randint(-10,10)
                # print(cc)
                img_hsv[:,:,0]=(img_hsv[:,:,0]+hue_x)
                img_hsv[:,:,0] =np.maximum(img_hsv[:,:,0],0)
                img_hsv[:,:,0] =np.minimum(img_hsv[:,:,0],180)#范围 0 ~180
                img=cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)

        # img_ = prewhiten(img_)
        img_ = img_.astype(np.float32)
        img_ = (img_-128.)/256.
        # cv2.namedWindow('image',1)
        # cv2.imshow('image',img_)
        # cv2.waitKey(1)
        # print(img_[0:10,0:10,0])
        img_ = img_.transpose(2, 0, 1)

        return img_,label_
