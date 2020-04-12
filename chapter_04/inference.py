#-*-coding:utf-8-*-
# date:2020-04-12
# Author: xiang li
# function: inference

import os
import argparse
import torch
import torch.nn as nn
from data_iter.datasets import letterbox
import numpy as np


import time
import datetime
import os
import math
from datetime import datetime
import cv2
import torch.nn.functional as F

from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=' Project Classification Test')
    parser.add_argument('--test_model', type=str, default = './model_s152_dir/model_epoch-450.pth',
        help = 'test_model') # 模型路径
    parser.add_argument('--model', type=str, default = 'resnet_152',
        help = 'model : resnet_18,resnet_34,resnet_50,resnet_101,resnet_152') # 模型类型
    parser.add_argument('--num_class', type=int , default = 200,
        help = 'num_class') #  分类类别个数
    parser.add_argument('--GPUS', type=str, default = '0',
        help = 'GPUS') # GPU选择
    parser.add_argument('--test_path', type=str, default = './example/',
        help = 'test_path') # 测试集路径
    parser.add_argument('--img_size', type=tuple , default = (224,224),
        help = 'img_size') # 输入模型图片尺寸
    parser.add_argument('--fix_res', type=bool , default = True,
        help = 'fix_resolution') # 输入模型样本图片是否保证图像分辨率的长宽比
    parser.add_argument('--vis', type=bool , default = True,
        help = 'vis') # 是否可视化图片

    print('\n/******************* {} ******************/\n'.format(parser.description))
    #--------------------------------------------------------------------------
    ops = parser.parse_args()# 解析添加参数
    #--------------------------------------------------------------------------
    print('----------------------------------')

    unparsed = vars(ops) # parse_args()方法的返回值为namespace，用vars()内建函数化为字典
    for key in unparsed.keys():
        print('{} : {}'.format(key,unparsed[key]))

    #---------------------------------------------------------------------------
    os.environ['CUDA_VISIBLE_DEVICES'] = ops.GPUS

    test_path =  ops.test_path # 测试图片文件夹路径
    num_classes = ops.num_class # 模型类别个数
    print('num_classes : ',num_classes)
    #---------------------------------------------------------------- 构建模型
    print('use model : %s'%(ops.model))

    if ops.model == 'resnet_18':
        model_=resnet18()
    elif ops.model == 'resnet_34':
        model_=resnet34()
    elif ops.model == 'resnet_50':
        model_=resnet50()
    elif ops.model == 'resnet_101':
        model_=resnet101()
    elif ops.model == 'resnet_152':
        model_=resnet152()
    else:
        print('error no the struct model : {}'.format(ops.model))

    num_ftrs = model_.fc.in_features
    model_.fc = nn.Linear(num_ftrs, num_classes)

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_cuda else "cpu")
    model_ = model_.to(device)
    model_.eval() # 设置为前向推断模式

    # print(model_)# 打印模型结构

    # 加载测试模型
    if os.access(ops.test_model,os.F_OK):# checkpoint
        chkpt = torch.load(ops.test_model, map_location=device)
        model_.load_state_dict(chkpt)
        print('load test model : {}'.format(ops.test_model))

    #---------------------------------------------------------------- 预测图片
    font = cv2.FONT_HERSHEY_SIMPLEX
    for file in os.listdir(ops.test_path):
        gt_label = file.split('_label_')[-1].strip('.jpg')
        print('------>>> {} - gt_label : {}'.format(file,gt_label))

        img = cv2.imread(ops.test_path + file)
        # 输入图片预处理
        if ops.fix_res:
            img_ = letterbox(img,size_=ops.img_size[0],mean_rgb = (128,128,128))
        else:
            img_ = cv2.resize(img, (ops.img_size[1],ops.img_size[0]), interpolation = cv2.INTER_CUBIC)
        if ops.vis:
            cv2.namedWindow('image',0)
            cv2.imshow('image',img_)
            cv2.waitKey(1)
        img_ = img_.astype(np.float32)
        img_ = (img_-128.)/256.

        img_ = img_.transpose(2, 0, 1)
        img_ = torch.from_numpy(img_)
        img_ = img_.unsqueeze_(0)

        if use_cuda:
            img_ = img_.cuda()  # (bs, 3, h, w)

        pre_ = model_(img_.float())

        outputs = F.softmax(pre_,dim = 1)
        outputs = outputs[0]

        output = outputs.cpu().detach().numpy()
        output = np.array(output)

        max_index = np.argmax(output)

        score_ = output[max_index]

        print('gt {} -- pre {} : {}'.format(gt_label,max_index,score_))
        show_str = 'gt {} - pre {} :{:.2f}'.format(gt_label,max_index,score_)
        cv2.putText(img,show_str,(3,img.shape[0]-10),font,0.45,(15,125,255),3)
        cv2.putText(img,show_str,(3,img.shape[0]-10),font,0.45,(225,155,55),1)

        cv2.namedWindow('image',0)
        cv2.imshow('image',img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    print('well done ')
