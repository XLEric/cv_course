#-*-coding:utf-8-*-
# date:2020-04-12
# Author: xiang li
# function: inference

import os
import argparse
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

def train_val_loss(ops):
    f = open(ops.file, encoding='utf-8')#读取 json文件
    data = json.load(f)
    f.close()

    # print(data)
    loss_train_list = []
    loss_val_list = []

    sort_loss_train_list = []
    sort_loss_val_list = []

    for key in data.keys():
        print(key,data[key])
        loss_train_list.append(data[key]['loss_train'])
        loss_val_list.append(data[key]['loss_val'])

        sort_loss_train_list.append([data[key]['loss_train'],int(key.split('_')[-1])])# 为了选取 top val model
        sort_loss_val_list.append([data[key]['loss_val'],int(key.split('_')[-1])])# 为了选取 top val model

    sort_loss_train_list = sorted(sort_loss_train_list, key=lambda x:[x[0]], reverse=False)
    sort_loss_val_list = sorted(sort_loss_val_list, key=lambda x:[x[0]], reverse=False)
    print(sort_loss_val_list)
    for i in range(len(sort_loss_val_list)):
        print('{}) {}'.format(i,sort_loss_val_list[i]))

    #绘图
    mpl.rcParams['font.sans-serif'] = u'SimHei'
    mpl.rcParams['axes.unicode_minus'] = False

    fig = plt.figure()

    plt.plot(range(len(loss_train_list)), loss_train_list, c='red',lw = 2,label = 'train' )
    plt.scatter(range(len(loss_train_list)), loss_train_list,s=25, c='b', marker = 'o')

    plt.plot(range(len(loss_val_list)), loss_val_list, c='blue',lw = 2,label = 'val' )
    plt.scatter(range(len(loss_val_list)), loss_val_list,s=25, c='b', marker = 'o')

    colors = ['red','orange','purple','green','pink',]
    for i in range(min(len(sort_loss_val_list),5)):
        plt.scatter(sort_loss_val_list[i][1],sort_loss_val_list[i][0],s=80, color=colors[i], marker = 'o',
            label = str(i+1)+' val loss (epoch-%s): %.6f'%(sort_loss_val_list[i][1],sort_loss_val_list[i][0]))
        plt.text(sort_loss_val_list[i][1],sort_loss_val_list[i][0]+0.1+0.12*i, u'%.6f'%(sort_loss_val_list[i][0]),size=13)
        x_ = [sort_loss_val_list[i][1],sort_loss_val_list[i][1]]
        y_ = [sort_loss_val_list[i][0],sort_loss_val_list[i][0]+0.1+0.12*i]
        plt.plot(x_,y_, c='red',lw = 1 )

    for i in range(min(len(sort_loss_train_list),5)):
        plt.scatter(sort_loss_train_list[i][1],sort_loss_train_list[i][0],s=80, color=colors[i], marker = 'o',
            label = str(i+1)+' train loss (epoch-%s) : %.9f'%(sort_loss_train_list[i][1],sort_loss_train_list[i][0]))
        plt.text(sort_loss_train_list[i][1],-(sort_loss_train_list[i][0]+0.2+0.12*i), u'%.9f'%(sort_loss_train_list[i][0]),size=13)
        x_ = [sort_loss_train_list[i][1],sort_loss_train_list[i][1]]
        y_ = [-sort_loss_train_list[i][0],-(sort_loss_train_list[i][0]+0.2+0.12*i)]
        plt.plot(x_,y_, c='green',lw = 1 )

    plt.xlabel('Epoch', fontsize=13)
    plt.ylabel('trainVal Loss', fontsize=13)
    plt.legend(loc='upper right', fancybox=True, framealpha=0.8, fontsize=12)
    plt.title(u'trainVal Loss - %s'%ops.model, fontsize=17)
    # 设置刻度间隔
    x_major_locator=MultipleLocator(5)
    y_major_locator=MultipleLocator(0.5)
    ax=plt.gca()#两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    # 设置刻度显示范围
    plt.xlim(0.,121)
    plt.ylim(-1.,5)
    # 保存绘图
    plt.savefig('trainVal_loss.jpg')
    plt.show()
    fig.tight_layout()
    fig.tight_layout()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=' trainval loss ')
    parser.add_argument('--file', type=str, default = './model_exp/2020-04-19_20-14-21/loss_epoch_trainval.json',
        help = 'file') # 分析文件路径
    parser.add_argument('--model', type=str, default = 'ResNet_101',
        help = 'model') # 模型类型

    print('\n/******************* {} ******************/\n'.format(parser.description))
    #--------------------------------------------------------------------------
    args = parser.parse_args()# 解析添加参数

    train_val_loss(ops= args)
