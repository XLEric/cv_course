#-*-coding:utf-8-*-
# date:2020-04-05
# Author: xiang li
import os
import cv2 # 加载OpenCV库
import argparse # 加载处理命令行参数库
import shutil
import json
from utils.common_utils import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--images_path', type=str, default = './datasets/CUB_200_2011/images/',
        help = 'images_path') # 添加图片路径
    parser.add_argument('--images_list', type=str, default = './datasets/CUB_200_2011/images.txt',
        help = 'images_list')# 添加图片信息文本路径
    parser.add_argument('--train_test_split', type=str, default = './datasets/CUB_200_2011/train_test_split.txt',
        help = 'train_test_split')# 添加训练集和测试集分割信息文本路径
    parser.add_argument('--bbox_msg', type=str, default = './datasets/CUB_200_2011/bounding_boxes.txt',
        help = 'bounding_boxes')# 添加图片边界框信息文件路径
    parser.add_argument('--train_datasets', type=str, default = './datasets/train_datasets/',
        help = 'train_datasets')# 添加训练集文件夹路径
    parser.add_argument('--test_datasets', type=str, default = './datasets/test_datasets/',
        help = 'test_datasets')# 添加测试集文件夹路径
    parser.add_argument('--make_cropdata', type=bool, default = True,
        help = 'make_bbox_datastes')# 添加是否crop之后的训练集
    parser.add_argument('--vis', type=bool, default = False,
        help = 'visualization')# 添加可视化标志位

    args = parser.parse_args()# 解析添加参数
    print('----------------------------------')


    unparsed = vars(args) # parse_args()方法的返回值为namespace，用vars()内建函数化为字典
    for key in unparsed.keys():
        print('{} : {}'.format(key,unparsed[key]))

    unparsed['m'] = 999
    fs = open('./make_data_train_test_msg.tr_param',"w",encoding='utf-8')
    json.dump(unparsed,fs,ensure_ascii=False,indent = 1)
    fs.close()

    mkdir_(args.train_datasets,flag_rm = True)# 创建训练集文件夹
    mkdir_(args.test_datasets,flag_rm = True)# 创建测试集文件夹

    imgs_list = open(args.images_list,'r',encoding = 'utf-8').readlines()# 图片列表
    print(len(imgs_list))
    train_test_split_list = open(args.train_test_split,'r',encoding = 'utf-8').readlines()# 训练和测试集分割列表
    print(len(train_test_split_list))
    bbox_msg_list = open(args.bbox_msg,'r',encoding = 'utf-8').readlines()# 训练和测试集分割列表
    print(len(train_test_split_list))

    train_cnt = 0
    test_cnt = 0
    img_mean_w = 0.# 统计crop图片 width 均值
    img_mean_h = 0.# 统计crop图片 height 均值
    for i in range(len(imgs_list)):
        image_id,image_name = imgs_list[i].strip('\n').split(' ')# 获取信息 ：图片id 、图片相对路径
        doc_class = image_name.split('/')[0]# 每一类文件夹
        _,is_train = train_test_split_list[i].strip('\n').split(' ')# 获取信息 ：图片id 、是否是训练图片标志位
        is_train = int(is_train)
        _,x,y,w,h = bbox_msg_list[i].strip('\n').split(' ')# 获取信息 ：图片id 、边界框
        x1,y1,x2,y2 = int(float(x)),int(float(y)),int(float(x) + float(w)),int(float(y) + float(h))
        print('[{}/{}] {} : is_train {} '.format(image_id,len(imgs_list),image_name,is_train))

        if is_train:# 训练集
            path_s = args.train_datasets
            train_cnt += 1
        else:# 测试集
            path_s = args.test_datasets
            test_cnt += 1

        mkdir_(path_s + doc_class)# 创建训练测试集的分类目录

        if args.make_cropdata:# 是否制作 crop 数据集
            img = cv2.imread(args.images_path + image_name)
            cv2.imwrite(path_s +image_name,img[y1:y2,x1:x2,:])
            img_mean_w += (x2-x1)
            img_mean_h += (y2-y1)
            print('w,h : ({},{})'.format(int(img_mean_w/(train_cnt+test_cnt)),int(img_mean_h/(train_cnt+test_cnt))))
        else:# 制作原图数据集
            shutil.copyfile(args.images_path + image_name,path_s +image_name )# 将原图片路径拷贝到相应的训练集和测试集类别文件夹

        if args.vis:
            img = cv2.imread(args.images_path + image_name)# 读取图片可视化

            bbox = x1,y1,x2,y2
            plot_box(bbox, img, color=(255,0,0), label='bird')

            cv2.namedWindow('img',0)#窗口大小可改变
            cv2.imshow('img',img)#显示旋转图片
            cv2.waitKey(1)

    print('train datasets len : {}'.format(train_cnt))
    print('test datasets len : {}'.format(test_cnt))
    if args.vis:
        cv2.destroyAllWindows()
