#-*-coding:utf-8-*-
# date:2020-05-07
# Author: xiang li

from __future__ import print_function
import argparse
import os
import sys
import time
sys.path.append('./')
import torch
import torch.nn as nn
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from models.mobilenetv2 import MobileNetV2

from thop import profile
from thop import clever_format # 增加可读性


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=' Retinaface Inferece')

    parser.add_argument('--network',  type=str, default='resnet_50',
        help='Backbone network : resnet_18,resnet_34,resnet_50,resnet_101,resnet_152,mobilenetv2')

    parser.add_argument('--input_shape', type=tuple , default = (1,3,224,224),
        help = 'input_shape') #
    parser.add_argument('--num_classes', type=int , default = 1000,
        help = 'num_classes') # 模型输入图片颜色偏置

    parser.add_argument('--GPUS', type=str, default = '0',
        help = 'GPUS') # GPU选择

    print('\n/******************* {} ******************/\n'.format(parser.description))
    #--------------------------------------------------------------------------
    ops = parser.parse_args()# 解析添加参数
    #--------------------------------------------------------------------------
    print('----------------------------------')

    unparsed = vars(ops) # parse_args()方法的返回值为namespace，用vars()内建函数化为字典
    for key in unparsed.keys():
        print('{} : {}'.format(key,unparsed[key]))

    os.environ['CUDA_VISIBLE_DEVICES'] = ops.GPUS
    #---------------------------------------------------------------- 构建 landmarks 模型
    if ops.network == 'resnet_18':
        model_=resnet18(num_classes=ops.num_classes, img_size=ops.input_shape[2])
    elif ops.network == 'resnet_34':
        model_=resnet34(num_classes=ops.num_classes, img_size=ops.input_shape[2])
    elif ops.network == 'resnet_50':
        model_=resnet50(num_classes=ops.num_classes, img_size=ops.input_shape[2])
    elif ops.network == 'resnet_101':
        model_=resnet101(num_classes=ops.num_classes, img_size=ops.input_shape[2])
    elif ops.network == 'resnet_152':
        model_=resnet152(num_classes=ops.num_classes, img_size=ops.input_shape[2])
    elif ops.network == 'mobilenetv2':
        model_=MobileNetV2(n_class =ops.num_classes, input_size=ops.input_shape[2])
    else:
        print('error no the struct model : {}'.format(ops.network))

    dummy_input = torch.randn(ops.input_shape)
    flops, params = profile(model_, inputs=(dummy_input, ))
    print('flops ： {} , params : {}'.format(flops, params))

    flops, params = clever_format([flops, params], "%.3f")
    print('flops ： {} , params : {}'.format(flops, params))

    # params = list(model_.parameters())
    # idx = 0
    # for i in params:
    #     idx += 1
    #     print('{}) :  {}'.format(idx,i))
    print('/********************* modules *******************/')
    op_dict = {}
    idx = 0
    for m in model_.modules():
        idx += 1
        if isinstance(m, nn.Conv2d):
            if 'Conv2d' not in op_dict.keys():
                op_dict['Conv2d'] = 1
            else:
                op_dict['Conv2d'] += 1
            print('{})  {}'.format(idx,m))
            pass
        elif isinstance(m, nn.BatchNorm2d):
            if 'BatchNorm2d' not in op_dict.keys():
                op_dict['BatchNorm2d'] = 1
            else:
                op_dict['BatchNorm2d'] += 1
            print('{})  {}'.format(idx,m))
            pass
        elif isinstance(m, nn.Linear):
            if 'Linear' not in op_dict.keys():
                op_dict['Linear'] = 1
            else:
                op_dict['Linear'] += 1
            print('{})  {}'.format(idx,m))
            pass
        elif isinstance(m, nn.Sequential):
            print('*******************{})  {}'.format(idx,m))
            for n in m:
                print('{})  {}'.format(idx,n))
                if 'Conv2d' not in op_dict.keys():
                    op_dict['Conv2d'] = 1
                else:
                    op_dict['Conv2d'] += 1
                if 'BatchNorm2d' not in op_dict.keys():
                    op_dict['BatchNorm2d'] = 1
                else:
                    op_dict['BatchNorm2d'] += 1
                if 'Linear' not in op_dict.keys():
                    op_dict['Linear'] = 1
                else:
                    op_dict['Linear'] += 1
                if 'ReLU6' not in op_dict.keys():
                    op_dict['ReLU6'] = 1
                else:
                    op_dict['ReLU6'] += 1
            pass
        elif isinstance(m, nn.ReLU6):
            print('{})  {}'.format(idx,m))
            if 'ReLU6' not in op_dict.keys():
                op_dict['ReLU6'] = 1
            else:
                op_dict['ReLU6'] += 1
            pass
        elif isinstance(m, nn.Module):
            print('{})  {}'.format(idx,m))
            for n in m.modules():
                if isinstance(n, nn.Conv2d):
                    print('{})  {}'.format(idx,n))
                    if 'Conv2d' not in op_dict.keys():
                        op_dict['Conv2d'] = 1
                    else:
                        op_dict['Conv2d'] += 1
                    if 'BatchNorm2d' not in op_dict.keys():
                        op_dict['BatchNorm2d'] = 1
                    else:
                        op_dict['BatchNorm2d'] += 1
                    if 'Linear' not in op_dict.keys():
                        op_dict['Linear'] = 1
                    else:
                        op_dict['Linear'] += 1
                    if 'ReLU6' not in op_dict.keys():
                        op_dict['ReLU6'] = 1
                    else:
                        op_dict['ReLU6'] += 1
                    pass
            pass

        else:
            print('{})  {}'.format(idx,m))
            pass

    print('\n/********************** {} ********************/\n'.format(ops.network))
    for key in op_dict.keys():
        print(' operation - {} : {}'.format(key,op_dict[key]))
