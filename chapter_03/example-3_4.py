#-*-coding:utf-8-*-
# date:2020-03-28
# Author: xiang li

import numpy as np # 加载 Numpy 库
import torch # 加载 Torch 库
'''
思考这一节的用法与实际项目的使用方式结合
'''
if __name__ == "__main__":
    x = torch.tensor([[1.,2.],[3.,4.]])
    y = torch.tensor([[5.,6.],[7.,8.]])
    print('x : ',x)
    print('y : ',y)
    print('x-y : ',x-y)# Tensor对应元素相减
    print('x+y : ',x+y)# Tensor对应元素相加
    print('x*y : ',x*y)# Tensor对应元素相乘
    print('x/y : ',x/y)# Tensor对应元素相除
    print('matmul(x,y) : ',torch.matmul(x,y))# 两个Tensor相乘

    print('-------------------------------')
    x = torch.tensor([9.,4.,25.])
    print('x : ',x)
    print('sqrt(x) : ',x.sqrt())# Tensor对应元素开方

    print('-------------------------------')
    x = torch.tensor([-9.,2.,-13.])
    print('x : ',x)
    print('abs(x) : ',torch.abs(x))# Tensor对应元素取绝对值

    print('-------------------------------')

    print('x : ',x)
    print(x**3.)# 对应元素求幂

    print('-------------------------------')
    # 三角函数
    x = torch.tensor(np.pi/3.)#60 度
    print('sin(x) : ',torch.sin(x))
    print('cos(x) : ',torch.cos(x))
    print('tan(x) : ',torch.tan(x))

    print('-------------------------------')
    # 反三角函数
    x = torch.tensor(0.866)
    print('asin(x) : ',torch.asin(x))
    print('acos(x) : ',torch.acos(x))
    print('atan(x) : ',torch.atan(x))

    print('-------------------------------')
    # 求和
    x = torch.tensor([[1.,2.],[3.,4.]])
    print('x : ',x)
    print('sum(x) : ',torch.sum(x))
    print('sum(x) (dim = 0) : ',torch.sum(x,dim = 0))# 按维度求和
    print('sum(x) (dim = 1) : ',torch.sum(x,dim = 1))# 按维度求和

    print('-------------------------------')
    # 求均值
    print('x : ',x)
    print('mean(x) : ',torch.mean(x))
    print('mean(x) (dim = 0) : ',torch.mean(x,dim = 0))# 按维度求均值
    print('mean(x) (dim = 1) : ',torch.mean(x,dim = 1))# 按维度求均值

    print('-------------------------------')
    # 求最大值
    print('x : ',x)
    print('max(x) : ',torch.max(x))
    print('max(x) (dim = 0) : ',torch.max(x,dim = 0))#返回值和对应索引
    print('max(x) (dim = 1) : ',torch.max(x,dim = 1))#返回值和对应索引

    print('-------------------------------')
    # 求最小值
    print('x : ',x)
    print('min(x) : ',torch.min(x))
    print('min(x) (dim = 0) : ',torch.min(x,dim = 0))#返回值和对应索引
    print('min(x) (dim = 1) : ',torch.min(x,dim = 1))#返回值和对应索引
