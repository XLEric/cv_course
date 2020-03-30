#-*-coding:utf-8-*-
# date:2020-03-28
# Author: xiang li

import numpy as np # 加载 Numpy 库
import torch # 加载 Torch 库
'''
思考这一节的用法与实际项目的使用方式结合
'''
if __name__ == "__main__":

    x = torch.tensor(3.1415)
    print(x.floor())# tensor 向上取整
    print(x.ceil())# tensor 向下取整
    print(x.trunc())# tensor 取整数部分
    print(x.frac())# tensor 取小数部分
    y = torch.tensor(3.4)
    z = torch.tensor(3.5)
    print(y.round(), z.round())# 对tensor四舍五入
