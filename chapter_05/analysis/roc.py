#-*-coding:utf-8-*-
# date:2020-04-12
# Author: xiang li
#function： ROC

import os
import argparse
import json
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import matplotlib as mpl
import matplotlib.pyplot as plt
import random

'''
方法1：每种类别下，都可以得到m个测试样本为该类别的概率（矩阵P中的列）。
所以，根据概率矩阵P和标签矩阵L中对应的每一列，可以计算出各个阈值下的假正例率（FPR）和真正例率（TPR），绘制出一条ROC曲线。
总共可以绘制出n条ROC曲线。最后对n条ROC曲线取平均，即可得到最终的ROC曲线。
方法2：单个测试样本：1）标签只由0和1组成，1的位置表明了它的类别（可对应二分类问题中的‘’正’’），0就表示其他类别（‘’负‘’）；
2）要是分类器对该测试样本分类正确，则该样本标签中1对应的位置在概率矩阵P中的值是其对应的概率值。
基于这两点，将标签矩阵L和概率矩阵P分别按行展开，转置后形成两列，这就得到了一个二分类的结果。
所以，此方法经过计算后可以直接得到最终的ROC曲线。
'''

def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color
def roc(ops):
    f = open(ops.file, encoding='utf-8')#读取 json文件
    data = json.load(f)
    f.close()

    data = data['roc_metrics']

    # print('{}'.format(data))

    scores = []
    labels = []
    for i in range(len(data)):
        score,label = data[i]
        scores.append(score)
        labels.append(label)

    scores = np.array(scores)
    labels = np.array(labels)

    labels_one_hot = label_binarize(labels, np.arange(ops.num_classes))  #装换为独热编码

    print(scores.shape,labels_one_hot.shape)

    metrics.roc_auc_score(labels_one_hot, scores, average='micro')#调用函数计算micro类型的AUC  : micro / macro

    fpr, tpr, thresholds = metrics.roc_curve(labels_one_hot.ravel(),scores.ravel())#首先将矩阵 labels_one_hot 和 scores 展开，然后计算假正例率FPR和真正例率TPR
    auc = metrics.auc(fpr, tpr)
    print('AUC：', auc)

    #绘图
    mpl.rcParams['font.sans-serif'] = u'SimHei'
    mpl.rcParams['axes.unicode_minus'] = False
    #FPR就是横坐标,TPR就是纵坐标
    plt.plot(fpr, tpr, c = 'r', lw = 2, alpha = 0.7, label = u'AUC=%.6f' % auc)
    plt.plot((0, 1), (0, 1), c = '#808080', lw = 1, ls = '--', alpha = 0.7)

    max_dst = 0.
    max_dst_x,max_dst_y = 0.,0.
    max_thr = 0.
    for i in range(len(thresholds)):
        y = tpr[i]
        x = fpr[i]
        if max_dst < (y-x):
            max_dst = (y-x)
            max_dst_x,max_dst_y = x,y
            max_thr = thresholds[i]


    plt.scatter(max_dst_x,max_dst_y,s=30, c='b', marker = 'o')
    plt.text(max_dst_x,max_dst_y, u'fpr-tpr(%.3f,%.3f) ,thr=%.5f'%(max_dst_y,max_dst_x,max_thr))

    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
    plt.title(u'ROC和AUC', fontsize=17)
    print(ops.file.replace('.json','_roc.jpg'))
    plt.savefig(ops.file.replace('.json','_roc.jpg'))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=' ROC ')
    parser.add_argument('--file', type=str, default = './roc_metrics_2020-04-15_11-42-06.json',
        help = 'file') # 分析文件路径
    parser.add_argument('--num_classes', type=int, default = 200,
        help = 'num_classes') # 分类类别

    print('\n/******************* {} ******************/\n'.format(parser.description))
    #--------------------------------------------------------------------------
    args = parser.parse_args()# 解析添加参数
    #--------------------------------------------------------------------------
    print('----------------------------------')

    unparsed = vars(args) # parse_args()方法的返回值为namespace，用vars()内建函数化为字典
    for key in unparsed.keys():
        print('{} : {}'.format(key,unparsed[key]))

    roc(args)
