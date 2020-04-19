#-*-coding:utf-8-*-
# date:2020-04-12
# Author: xiang li
# function: inference

import os
import argparse
import json

def train_val_loss(ops):
    f = open(ops.file, encoding='utf-8')#读取 json文件
    data = json.load(f)
    f.close()

    print(data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=' trainval loss ')
    parser.add_argument('--file', type=str, default = './model_exp/2020-04-19_18-31-29/loss_epoch_trainval.json',
        help = 'file') # 分析文件路径


    print('\n/******************* {} ******************/\n'.format(parser.description))
    #--------------------------------------------------------------------------
    args = parser.parse_args()# 解析添加参数

    train_val_loss(ops= args)
