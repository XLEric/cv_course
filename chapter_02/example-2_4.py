#-*-coding:utf-8-*-
# date:2020-03-28
# Author: xiang li
# function: rgb2gray & resize

import cv2 #导入OpenCV数据库
if __name__ == "__main__":
    img = cv2.imread('./datasets/000000003671.jpg')# 读取图片

    print('image shape : ',img.shape)#打印图片尺寸
    cv2.imshow('image',img) # 显示图片
    cv2.waitKey(0)# 等待按键按下

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)# BGR图像转灰度图像

    cv2.imshow('gray',gray) # 显示图片
    cv2.waitKey(0)# 等待按键按下


    img =cv2.resize(img, (256,256))#图像缩放为256*256尺寸
    print('image shape : ',img.shape)#打印图片尺寸
    cv2.imshow('image_resize',img) # 显示图片
    cv2.waitKey(0)# 等待按键按下

    cv2.destroyAllWindows()# 销毁图片显示窗口
