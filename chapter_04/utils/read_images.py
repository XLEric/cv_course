#-*-coding:utf-8-*-
# date:2020-04-05
# Author: xiang li

import cv2 # 加载OpenCV库
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--images_path', type=str, default = './datasets/CUB_200_2011/images/',
        help = 'images_path') # 添加图片路径
    parser.add_argument('--images_list', type=str, default = './datasets/CUB_200_2011/images.txt',
        help = 'images_list')# # 添加图片信息文本路径

    args = parser.parse_args()# 解析添加参数
    print(args.images_path)
    print(args.images_list)

    imgs_list = open(args.images_list,'r',encoding = 'utf-8').readlines()
    print(len(imgs_list))

    for i in range(len(imgs_list)):
        image_id,image_name = imgs_list[i].strip('\n').split(' ')# 获取信息 ：图片id 、图片相对路径
        print('{}) {}'.format(image_id,image_name))
        img = cv2.imread(args.images_path + image_name)# 读取图片

        cv2.namedWindow('img',0)#窗口大小可改变
        cv2.imshow('img',img)#显示旋转图片
        cv2.waitKey(1)

    cv2.destroyAllWindows()
