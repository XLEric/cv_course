#-*-coding:utf-8-*-
# date:2020-03-28
# Author: xiang li
# function: image crop

import cv2
if __name__ == "__main__":
    img = cv2.imread('./datasets/000000003671.jpg')
    print('iamge shape : ',img.shape)
    x1,y1 = 100,150 # 裁剪部分的左上坐标
    x2,y2 = 550,400 # 裁剪部分的右下坐标
    img_crop = img[y1:y2,x1:x2,:]# 裁剪图片
    cv2.imshow('image',img)#显示原图
    cv2.namedWindow('crop',0)#窗口大小可改变
    cv2.imshow('crop',img_crop)#显示裁剪后的图片
    cv2.waitKey(0)
