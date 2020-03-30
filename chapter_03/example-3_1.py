#-*-coding:utf-8-*-
# date:2020-03-28
# Author: xiang li

import torch # 加载torch库
import numpy as np # 加载Numpy库
if __name__ == "__main__":
    print(torch.__version__)# 查看 torch 版本
    print('-----------------------')
    y = torch.rand(2,3)# 随机矩阵
    print(y)
    print(y.size())
    print('-----------------------')
    print(torch.zeros(2,2))#全0矩阵
    print('-----------------------')
    print(torch.ones(2,2))#全1矩阵
    print('-----------------------')
    print(torch.eye(3,3))# 单位矩阵
    print('-----------------------')
    print(torch.rand_like(input = y, dtype = torch.double))# 输出和input矩阵相同size的随机矩阵
