#-*-coding:utf-8-*-
# date:2020-03-28
# Author: xiang li
# function: image brightness contrast

import cv2 # 加载OpenCV库
import numpy as np # 加载Numpy库
if __name__ == "__main__":
    img = cv2.imread('./datasets/000000003671.jpg')# 读取图片
    a = 1.25 # 对比度值
    b = 20 # 亮度变化值
    img_ = img*a + b # 对比度亮度变化计算
    img_[img_>255] = 255 # 限定像素值不大于255
    img_[img_<0] = 0 # 限定像素值不小于0
    img_= img_.astype(np.uint8)# 图像数据强制转化为 uint8
    cv2.imshow('image',img)#显示原图
    cv2.imshow('image_',img_)#显示改变对比度和亮度的图片
    cv2.waitKey(0)
