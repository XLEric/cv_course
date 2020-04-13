#-*-coding:utf-8-*-
# date:2020-03-28
# Author: xiang li
# function: image rotation

import cv2 # 加载OpenCV库
import numpy as np # 加载Numpy库
if __name__ == "__main__":
    img = cv2.imread('./datasets/000000003671.jpg')# 读取图片
    angle = 45 # 旋转角度，设为45°
    cx ,cy = int(img.shape[1]/2),int(img.shape[0]/2)# 定义旋转中心
    borderValue = 0 # 旋转空缺部分的默认值，设为0
    (h , w) = img.shape[:2] # 图像的高和宽
    #计算旋转矩阵
    M = cv2.getRotationMatrix2D((cx , cy) , -angle , 1.0)
    cos = np.abs(M[0 , 0])
    sin = np.abs(M[0 , 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0 , 2] += int(0.5 * nW) - cx
    M[1 , 2] += int(0.5 * nH) - cy
    #更新旋转图片
    img_rot = cv2.warpAffine(img , M , (nW , nH),borderValue=borderValue)
    cv2.imshow('image',img)#显示原图
    cv2.namedWindow('img_rot',0)#窗口大小可改变
    cv2.imshow('img_rot',img_rot)#显示旋转图片
    cv2.waitKey(0)
