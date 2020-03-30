#-*-coding:utf-8-*-
# date:2020-03-28
# Author: xiang li

import numpy as np # 加载 Numpy 库
import torch # 加载 Torch 库
# 数据转换 ： numpy -》 cpu tensor -》 gpu tensor -》 numpy
if __name__ == "__main__":
    x=np.array([1,9])# 构建整型数组
    print(type(x))# 打印数组类型
    x_tensor=torch.from_numpy(x)# numpy转化为tensor
    print('x_tensor cpu : ',x_tensor)# 打印tensor
    print(x_tensor.type())# 打印tensor数据类型
    x_tensor = x_tensor.float()# tensor 的数据类型从整型转化为浮点型
    print(x_tensor.type())# 打印tensor数据类型
    print('x_tensor cpu : ',x_tensor)# 打印tensor

    print('----------------------------------------------')
    if torch.cuda.is_available():# 判断 GPU cuda 是否可用
        x_gpu_tensor = x_tensor.cuda()# CPU tensor转化为 GPU tensor
        print(x_gpu_tensor.type())# 打印GPU tensor数据类型
        print('x_gpu_tensor : ',x_gpu_tensor)

        x_cpu_tensor = x_gpu_tensor.cpu()# GPU tensor转化为 CPU tensor
        print('x_cpu_tensor : ',x_cpu_tensor)
        x_numpy = x_cpu_tensor.numpy()# CPU tensor 转化为 numpy
        print('x_numpy : ',x_numpy)
