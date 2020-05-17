#-*-coding:utf-8-*-
# date:2019-12-10
# Author: xiang li
# function: image agu

import os
import numpy as np
import random
import cv2
from imgaug import augmenters as iaa
import math

# random.seed(666)

def imag_aug_iaa_fun(imgn,idx=0):
    img_aug_list=[]
    img_aug_list.append(imgn.copy())

    if idx == 0:#单一方式增强
        seq = iaa.Sequential([iaa.Sharpen(alpha=(0.0, 0.45), lightness=(0.65, 1.35))])
        # print('-------------------->>> imgaug 0 : Sharpen')
    elif idx == 1:
        seq = iaa.Sequential([iaa.AverageBlur(k=(2))])# blur image using local means with kernel sizes between 2 and 4
        # print('-------------------->>> imgaug 1 : AverageBlur')
    elif idx == 2:
        seq = iaa.Sequential([iaa.MedianBlur(k=(3))])# blur image using local means with kernel sizes between 3 and 5
        # print('-------------------->>> imgaug 2 : MedianBlur')
    elif idx == 3:
        seq = iaa.Sequential([iaa.GaussianBlur((0.0, 0.55))])
        # print('-------------------->>> imgaug 3 : GaussianBlur')
    elif idx == 4:
        seq = iaa.Sequential([iaa.ContrastNormalization((0.90, 1.10))])
        # print('-------------------->>> imgaug 4 : ContrastNormalization') #  对比度
    elif idx == 5:
        seq = iaa.Sequential([iaa.Add((-55, 55))])
        # print('-------------------->>> imgaug 5 : Add')
    elif idx == 6:
        seq = iaa.Sequential([iaa.AddToHueAndSaturation((-10, 10),per_channel=True)])
        # print('-------------------->>> imgaug 6 : AddToHueAndSaturation')
    elif idx == 7:
        # seq = iaa.Sequential([iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.02*255), per_channel=False, name=None, deterministic=False, random_state=None)])
        seq = iaa.Sequential([iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1*255), per_channel=False, name=None, deterministic=False, random_state=None)])
        # print('-------------------->>> imgaug 7 : AdditiveGaussianNoise')
    elif idx == 8:#复合增强方式
        # print(' *** 复合增强方式')
        # print('-------------------->>> 符合增强')
        seq = iaa.Sequential([
            iaa.Sharpen(alpha=(0.0, 0.05), lightness=(0.9, 1.1)),
            iaa.GaussianBlur((0, 0.8)),
            iaa.ContrastNormalization((0.9, 1.1)),
            iaa.Add((-5, 5)),
            iaa.AddToHueAndSaturation((-5, 5)),
        ])
    images_aug = seq.augment_images(img_aug_list)
    return images_aug[0].copy()

def img_agu_contrast(img_, a, b):
    # print('contrast_img : a %s b %s'%(a,b))
    img_contrast_ = np.float(a)*img_ + b
    img_contrast_[img_contrast_>255] = 255
    img_contrast_[img_contrast_<0] = 0
    return img_contrast_.astype(np.uint8)

def img_agu_random_color_channel(img_):#颜色通道变换
    # print('----->> random_image_color_channel')
    idx = [j for j in range(3)]
    random.shuffle(idx)
    # print('random_image_color_channel : ',idx)
    img_c_ = np.zeros([img_.shape[0],img_.shape[1],img_.shape[2]], dtype = np.uint8)
    img_c_[:,:,0] = img_[:,:,idx[0]]
    img_c_[:,:,1] = img_[:,:,idx[1]]
    img_c_[:,:,2] = img_[:,:,idx[2]]
    return img_c_

def img_agu_hue(img_):
    img_hsv_=cv2.cvtColor(img_,cv2.COLOR_BGR2HSV)
    hue_x = random.randint(-10,10)
    img_hsv_[:,:,0]=(img_hsv_[:,:,0]+hue_x)
    img_hsv_[:,:,0] =np.maximum(img_hsv_[:,:,0],0)
    img_hsv_[:,:,0] =np.minimum(img_hsv_[:,:,0],180)#范围 0 ~180
    img_hsv_=cv2.cvtColor(img_hsv_,cv2.COLOR_HSV2BGR)
    return img_hsv_
def img_agu_rot_offset(image , angle , cx , cy,borderValue=0):
    '''
    图像旋转
    :param image:
    :param angle:
    :return: 返回旋转后的图像以及旋转矩阵
    '''
    # print('angle %s , cx %s , cy %s'%(angle , cx , cy))
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
    return cv2.warpAffine(image , M , (nW , nH),borderValue=borderValue) , M

def img_agu_resize(img_,size_=(128,128)):
    resize_model = [cv2.INTER_LINEAR,cv2.INTER_CUBIC,cv2.INTER_NEAREST,cv2.INTER_AREA,cv2.INTER_LANCZOS4]
    # img_resize_ = cv2.resize(img_, down_size, interpolation = random.randint(0,4))
    img_resize_ = cv2.resize(img_, size_, interpolation = random.randint(0,4))
    return img_resize_

def img_agu_flip(img_):
    if (random.randint(0,1)==1 ):#水平翻转
        img_flip_ = cv2.flip(img_.copy(),1)
    else:
        img_flip_ = img_.copy()
    return img_flip_

def img_agu_crop(img_):
    scale_ = int(min(img_.shape[0],img_.shape[1])/4)
    x1 = max(0,random.randint(0,scale_))
    y1 = max(0,random.randint(0,scale_))
    x2 = min(img_.shape[1]-1,img_.shape[1] - random.randint(0,scale_))
    y2 = min(img_.shape[0]-1,img_.shape[1] - random.randint(0,scale_))
    # print(img_.shape,'-crop- : ',x1,y1,x2,y2)
    img_crop_ = img_[y1:y2,x1:x2,:]
    return img_crop_

def img_agu_fix_size_no_deformation(img_,size_=416,mean_rgb = (128,128,128)):

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

def img_agu_scale(img_,list_x,label_num_dict,size = 300,mean_rgb=(128,128,128),mulriple = 2.5):

    max_size = max(img_.shape[0],img_.shape[1])
    scale = random.uniform(1,mulriple)
    max_size = int(scale*max_size)
    pad_x_ = max_size-img_.shape[1]
    pad_y_ = max_size-img_.shape[0]
    left_ = random.randint(0,pad_x_)
    right_ = pad_x_ - left_
    top_ = random.randint(0,pad_y_)
    bottom_ = pad_y_ - top_

    # print('pad_x,pad_y,left_,right_,top_, bottom_',pad_x_,pad_y_,left_,right_,top_, bottom_)

    img_a = cv2.copyMakeBorder(img_, top_, bottom_, left_, right_, cv2.BORDER_CONSTANT, value=mean_rgb)  # padded square
    # img_a = cv2.resize(img_a, (size,size), interpolation=cv2.INTER_LINEAR)

    flag_flip = False

    # if random.randint(0,10)==1:
    #     img_a = cv2.flip(img_a, 1)# 0上下翻转 ，-1，上下+左右翻转 ，1左右翻转
    #     flag_flip = True

    # target
    target_ = []
    for j in range(len(list_x)):
        label_,bbox_ = list_x[j]
        # 归一化坐标
        x1,y1,x2,y2 = float(bbox_[0]+left_)/float(max_size),float(bbox_[1]+top_)/float(max_size),\
        float(bbox_[2]+left_)/float(max_size),float(bbox_[3]+top_)/float(max_size)

        #----------------------尺寸控制
        if np.minimum(np.abs(x2-x1),np.abs(y2-y1))<0.015:
            target_.append((0,0,0,0,0))
        else:
            if flag_flip == True:
                target_.append((1.-x2,y1,1.-x1,y2,int(label_num_dict[label_]))) # label_num_dict[label_] : label 的标签转为数字
            else:
                target_.append((x1,y1,x2,y2,int(label_num_dict[label_])))

    return img_a,target_

def img_agu_mixup(img_a,img_b):
    # INTER_AREA  INTER_CUBIC INTER_LINEAR
    img_b = cv2.resize(img_b, (img_a.shape[1],img_a.shape[0]), interpolation=cv2.INTER_LINEAR)
    alfa = 1.- float(random.randint(150,300))/1000.
    img_agu = cv2.addWeighted(img_a, alfa, img_b, (1.-alfa), 0)
    return img_agu
def img_agu_channel_same(img_):
    img_a = np.zeros(img_.shape, dtype = np.uint8)
    gray = cv2.cvtColor(img_,cv2.COLOR_RGB2GRAY)
    img_a[:,:,0] =gray
    img_a[:,:,1] =gray
    img_a[:,:,2] =gray

    return img_a

def img_agu_zoom(img_):
    zoom_range = int(min(img_.shape[0],img_.shape[1])/3)
    zoom_step = random.randint(0,zoom_range)
    img_a = img_[zoom_step:(img_.shape[0]-zoom_step),zoom_step:(img_.shape[1]-zoom_step),:]
    img_a
    return img_a
def img_agu_cover(img_ ,color = (0,0,0)):

    cx = random.randint(0,img_.shape[1]-1)
    cy = random.randint(0,img_.shape[0]-1)

    w_c = random.randint(30,100)
    h_c = random.randint(30,100)

    c1 = cx-w_c,cy-h_c
    c2 = cx+w_c,cy+h_c
    cv2.rectangle(img_, c1, c2, color, -1)

    return img_


if __name__ == "__main__":
    path_ = './images/'
    while True:
        for file_ in os.listdir(path_):
            if not('.jpg' in file_ or '.png' not in file_):
                continue
            print(file_)
            #------------------------------------------------>>
            img_ = cv2.imread(path_ + file_)
            #------------------------------------------------>>
            a = np.float(random.randint(50,150))/100.
            img_contrast_ = img_agu_contrast(img_, a, random.randint(-30,30))
            #------------------------------------------------>>
            img_c_ = img_agu_random_color_channel(img_)
            #------------------------------------------------>>
            img_hsv_ = img_agu_hue(img_)
            #------------------------------------------------>>
            scale_ = int(min(10,min(img_.shape[1],img_.shape[0])/8))
            cx = img_.shape[1]/2+random.randint(-scale_,scale_)
            cy = img_.shape[0]/2+random.randint(-scale_,scale_)
            angle = random.randint(-45,45)

            img_rot_,_ = img_agu_rot_offset(img_ , angle , int(cx) , int(cy),borderValue=(128,128,128))
            #------------------------------------------------>>
            img_resize_ = img_agu_resize(img_)

            #------------------------------------------------->>
            img_flip_ = img_agu_flip(img_)

            #------------------------------------------------->>
            img_iaa_ = imag_aug_iaa_fun(img_,idx=random.randint(0,8))

            #------------------------------------------------->>
            img_crop_ = img_agu_crop(img_)
            #------------------------------------------------->>
            img_fix_size_ = img_agu_fix_size_no_deformation(img_,size_=416,mean_rgb = (128,128,128))
            #------------------------------------------------->>
            img_agu_scale_ = img_agu_scale(img_.copy())
            #------------------------------------------------->>
            while True:

                random_files = random.sample(os.listdir(path_), 1)
                if file_ != random_files[0]:
                    img_b_ = cv2.imread(path_ + random_files[0])
                    img_mixup_ = img_agu_mixup(img_,img_b_)
                    break
            # print('mix_up',file_,random_files[0])
            #------------------------------------------------->>
            img_channel_s_ = img_agu_channel_same(img_)
            #------------------------------------------------->>
            img_zoom_ = img_agu_zoom(img_)
            #------------------------------------------------->>
            img_cover_ = img_agu_cover(img_.copy())

            cv2.namedWindow('image',0)
            cv2.imshow('image',img_)
            cv2.namedWindow('img_contrast',0)
            cv2.imshow('img_contrast',img_contrast_)
            cv2.namedWindow('img_random_color',0)
            cv2.imshow('img_random_color',img_c_)
            cv2.namedWindow('img_hsv',0)
            cv2.imshow('img_hsv',img_hsv_)
            cv2.namedWindow('img_rot',0)
            cv2.imshow('img_rot',img_rot_)
            cv2.namedWindow('img_resize',0)
            cv2.imshow('img_resize',img_resize_)
            cv2.namedWindow('img_flip',0)
            cv2.imshow('img_flip',img_flip_)
            cv2.namedWindow('img_iaa',0)
            cv2.imshow('img_iaa',img_iaa_)
            cv2.namedWindow('img_crop',0)
            cv2.imshow('img_crop',img_crop_)
            cv2.namedWindow('img_fix_size',0)
            cv2.imshow('img_fix_size',img_fix_size_)
            cv2.namedWindow('img_agu_scale',0)
            cv2.imshow('img_agu_scale',img_agu_scale_)
            cv2.namedWindow('img_mixup',0)
            cv2.imshow('img_mixup',img_mixup_)
            cv2.namedWindow('img_channel_s',0)
            cv2.imshow('img_channel_s',img_channel_s_)
            cv2.namedWindow('img_zoom',0)
            cv2.imshow('img_zoom',img_zoom_)
            cv2.namedWindow('img_agu_cover',0)
            cv2.imshow('img_agu_cover',img_cover_)





            if cv2.waitKey(500) == 27:
                flag_break = True
                break
        if cv2.waitKey(500) == 27:
            flag_break = True
            break

    cv2.destroyAllWindows()
