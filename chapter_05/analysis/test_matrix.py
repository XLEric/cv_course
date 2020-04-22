#-*-coding:utf-8-*-
# date:2020-04-12
# Author: xiang li
# function: test metrics

import os
import sys
sys.path.append('./')
import argparse
import torch
import torch.nn as nn
import numpy as np
from data_iter.datasets import letterbox
import time
import datetime
import math
from datetime import datetime
import cv2
import torch.nn.functional as F
import json
from loss.loss import FocalLoss
from utils.common_utils import *

from models.resnet import resnet50, resnet18, resnet34, resnet101, resnet152
def test(ops,timestamp):
    os.environ['CUDA_VISIBLE_DEVICES'] = ops.GPUS

    test_path =  ops.test_path
    #---------------------------------------------------------------- 构建模型
    print('use model : %s'%(ops.model))

    if ops.model == 'resnet_18':
        model_=resnet18(num_classes=ops.num_classes, img_size=ops.img_size[0])
    elif ops.model == 'resnet_34':
        model_=resnet34(num_classes=ops.num_classes, img_size=ops.img_size[0])
    elif ops.model == 'resnet_50':
        model_=resnet50(num_classes=ops.num_classes, img_size=ops.img_size[0])
    elif ops.model == 'resnet_101':
        model_=resnet101(num_classes=ops.num_classes, img_size=ops.img_size[0])
    elif ops.model == 'resnet_152':
        model_=resnet152(num_classes=ops.num_classes, img_size=ops.img_size[0])
    else:
        print('error no the struct model : {}'.format(ops.model))

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_cuda else "cpu")
    model_ = model_.to(device)
    model_.eval()

    # print(model_)# 打印模型结构

    # 加载测试模型
    if os.access(ops.test_model,os.F_OK):# checkpoint
        chkpt = torch.load(ops.test_model, map_location=device)
        model_.load_state_dict(chkpt)
        print('load test model : {}'.format(ops.test_model))

    dict_metrics = {}

    for idx,doc in enumerate(sorted(os.listdir(ops.test_path), key=lambda x:int(x.split('.')[0]), reverse=False)):
        path_doc = ops.test_path + doc + '/'

        if doc not in dict_metrics.keys():
            dict_metrics[idx] = {}
            dict_metrics[idx]['name'] = doc
            dict_metrics[idx]['pre_sum'] = 0.
            dict_metrics[idx]['pre_right'] = 0.
            dict_metrics[idx]['gt_sum'] = len(os.listdir(path_doc))

    roc_metrics_dict = {}
    roc_metrics_dict['roc_metrics'] = []
    for idx,doc in enumerate(sorted(os.listdir(ops.test_path), key=lambda x:int(x.split('.')[0]), reverse=False)):

        gt_label = idx
        print('{}) {} - gt_label : {}'.format(idx,doc,gt_label))
        path_doc = ops.test_path + doc + '/'

        for file in os.listdir(path_doc):
            img_ = cv2.imread(path_doc + file)

            if ops.fix_res:
                img_ = letterbox(img_,size_=ops.img_size[0],mean_rgb = (128,128,128))
            else:
                img_ = cv2.resize(img_, (ops.img_size[1],ops.img_size[0]), interpolation = cv2.INTER_CUBIC)
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
            roc_metrics_dict['roc_metrics'].append((output,gt_label))
            output = np.array(output)

            max_index = np.argmax(output)

            score_ = output[max_index]

            pre_str = str(idx) + ')  '+ ops.model + ' - score : '+str(score_)+' ---- gt label : '+str(gt_label)+' <-> pre label : '+str(max_index)
            print(pre_str)


            dict_metrics[max_index]['pre_sum'] += 1.

            if gt_label == max_index:
                dict_metrics[max_index]['pre_right'] += 1.

    #---------------------------------------------------------------------------
    fs = open('roc_metrics_{}.json'.format(timestamp),"w",encoding='utf-8')
    json.dump(roc_metrics_dict,fs,ensure_ascii=False,indent = 1,cls = JSON_Encoder)
    fs.close()

    MAP = 0.
    MRECALL = 0.

    try:
        for key in dict_metrics.keys():
            if dict_metrics[key]['pre_sum'] > 0:
                print('{}  Precision {} ReCall {}'.format(dict_metrics[key]['name'],
                    dict_metrics[key]['pre_right']/dict_metrics[key]['pre_sum'],
                    dict_metrics[key]['pre_right']/dict_metrics[key]['gt_sum']))
                MAP += (dict_metrics[key]['pre_right']/dict_metrics[key]['pre_sum'])
            MRECALL += (dict_metrics[key]['pre_right']/dict_metrics[key]['gt_sum'])

        dict_metrics['MAP'] = MAP/200.
        dict_metrics['MRECALL'] = MRECALL/200.
        print('MAP : {} , MRECALL : {}'.format(dict_metrics['MAP'],dict_metrics['MRECALL']))
    except Exception as e:
        print('Exception : ',e) # 打印异常
        print('Exception  file : ', e.__traceback__.tb_frame.f_globals['__file__'])# 发生异常所在的文件
        print('Exception  line : ', e.__traceback__.tb_lineno)# 发生异常所在的行数

    return dict_metrics

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=' Project Classification Test')
    parser.add_argument('--test_model', type=str, default = './model_exp/2020-04-20_14-42-25/model_epoch-123.pth',
        help = 'test_model') # 模型路径
    parser.add_argument('--model', type=str, default = 'resnet_101',
        help = 'model : resnet_18,resnet_34,resnet_50,resnet_101,resnet_152') # 模型类型
    parser.add_argument('--num_classes', type=int , default = 200,
        help = 'num_classes') #  分类类别个数
    parser.add_argument('--GPUS', type=str, default = '0',
        help = 'GPUS') # GPU选择
    parser.add_argument('--test_path', type=str, default = './datasets/test_datasets/',
        help = 'test_path') # 测试集路径
    parser.add_argument('--img_size', type=tuple , default = (256,256),
        help = 'img_size') # 输入模型图片尺寸
    parser.add_argument('--fix_res', type=bool , default = True,
        help = 'fix_resolution') # 输入模型样本图片是否保证图像分辨率的长宽比
    parser.add_argument('--vis', type=bool , default = False,
        help = 'vis') # 是否可视化图片

    print('\n/******************* {} ******************/\n'.format(parser.description))
    #--------------------------------------------------------------------------
    args = parser.parse_args()# 解析添加参数
    #--------------------------------------------------------------------------
    print('----------------------------------')

    unparsed = vars(args) # parse_args()方法的返回值为namespace，用vars()内建函数化为字典
    for key in unparsed.keys():
        print('{} : {}'.format(key,unparsed[key]))

    loc_time = time.localtime()

    unparsed['time'] = time.strftime("%Y-%m-%d %H:%M:%S", loc_time)

    loc_time_str = time.strftime("%Y-%m-%d_%H-%M-%S", loc_time)

    with torch.no_grad():
        dict_metrics = test(ops = args,timestamp = loc_time_str)

    unparsed['metrics'] = dict_metrics
    fs = open('test_{}.json'.format(loc_time_str),"w",encoding='utf-8')
    json.dump(unparsed,fs,ensure_ascii=False,indent = 1)
    fs.close()

    print('well done ')
