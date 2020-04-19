#-*-coding:utf-8-*-
# date:2020-04-18
# Author: xiang li
#function： mutilti label P-R

import os
import argparse
import json
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import matplotlib as mpl
import matplotlib.pyplot as plt
import random

def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color

def p_r(ops):
    f = open(ops.file, encoding='utf-8')#读取 json文件
    data = json.load(f)
    f.close()

    data = data['roc_metrics']

    # print('{}'.format(data))

    scores = []
    labels = []

    P_R_dict = {} # 记录 每一类的正反例子
    P_R_num_dict = {} # 记录 每一类的样本个数
    P_R_chart_dict = {} # 每一类的 precision / recall 对应表
    P_R_curve_dict = {} # 每一类的 P-R曲线
    #-------------------------------------------  创建每一类的字典
    for i in range(len(data)):
        scores,label = data[i]
        if label not in P_R_dict.keys():
            P_R_dict[label] = []
            P_R_chart_dict[label] = []
            P_R_curve_dict[label] = []
            P_R_num_dict[label] = 0
        P_R_num_dict[label] += 1

    for key in P_R_dict.keys():
        print('----------------->>>label {} len : {} '.format(key,P_R_num_dict[label]))
        for i in range(len(data)):
            scores,label = data[i]
            score = scores[key]
            max_index = np.argmax(scores)
            if key == label:# 多标签分类
                flag = 'p' # 正例
            else:
                flag = 'n' # 反例
            P_R_dict[key].append([flag,score])

    # 对每一类序列 score 进行排序
    for key in P_R_dict.keys():
        P_R_dict[key] = sorted(P_R_dict[key], key=lambda x: x[1],reverse = True)

    # 获得 P_R_chart
    for key in P_R_dict.keys():
        print('---------------------------->>> P_R_chart - label {}'.format(key))
        recall_num = 0
        pre_r_num = 0
        pre_num = 0
        for i in range(len(P_R_dict[key])):
            flag,score = P_R_dict[key][i]
            pre_num += 1
            if flag == 'p':
                pre_r_num += 1
                recall_num += 1
                print(' recall_num : {} ,pre_num : {} , recall : {} ,precision : {} '.format(
                    recall_num,
                    pre_num,
                    recall_num/P_R_num_dict[key],
                    pre_r_num/pre_num)
                    )
                P_R_chart_dict[key].append([recall_num/P_R_num_dict[key],pre_r_num/pre_num])


    # P-R 曲线 recall 坐标刻度
    recall_list = np.arange(0.0,1.01,0.01)
    print('recall_list len : ',len(recall_list))
    print('recall_list : ',recall_list)


    for key in P_R_chart_dict.keys():# 每一类别 P-R 表
        print('-------------->> get label {} P_R_curve'.format(key))
        for i in range(len(recall_list)):# recall 刻度
            r_ = recall_list[i]
            p_ = 1.
            for j in range(len(P_R_chart_dict[key])):
                recall_value,precision_value = P_R_chart_dict[key][j]
                if recall_list[i] >= recall_value:
                    # r_ = recall_value
                    p_ = precision_value
            P_R_curve_dict[key].append([r_,p_])

        # if True:
            # fig = plt.figure()
            #绘图
            # mpl.rcParams['font.sans-serif'] = u'SimHei'
            # mpl.rcParams['axes.unicode_minus'] = False
            # #FPR就是横坐标,TPR就是纵坐标
            #
            # curve_recall = [P_R_curve_dict[key][kk][0] for kk in range(len(P_R_curve_dict[key]))]
            # curve_precision = [P_R_curve_dict[key][kk][1] for kk in range(len(P_R_curve_dict[key]))]
            # print('R-P:',P_R_curve_dict[key])
            # print('x',curve_recall)
            # print('y',curve_precision)
    #         plt.plot(curve_recall,curve_precision, color=randomcolor(), lw = 2, alpha = 0.7)
    #
    #         plt.xlim((-0.01, 1.02))
    #         plt.ylim((-0.01, 1.02))
    #         plt.xticks(np.arange(0, 1.1, 0.1))
    #         plt.yticks(np.arange(0, 1.1, 0.1))
    #         plt.xlabel('Recall', fontsize=13)
    #         plt.ylabel('Precision', fontsize=13)
    #         plt.grid(b=True, ls=':')
    #         # plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
    #         plt.title(u'P-R_label %d'%(key), fontsize=17)
    # plt.show()

            # fig.tight_layout()
            # fig.tight_layout()

    ave_p_r_curve = []
    MAP = []
    MRECALL = []
    for i in range(len(recall_list)):
        recall_mean = []
        precision_mean = []
        for key in P_R_curve_dict.keys():
            recall_mean.append(P_R_curve_dict[key][i][0]) # recall
            precision_mean.append(P_R_curve_dict[key][i][1]) # precision
        ave_p_r_curve.append([np.mean(recall_mean),np.mean(precision_mean)])
        MAP.append(np.mean(precision_mean))
        MRECALL.append(np.mean(recall_mean))

    #绘图
    fig = plt.figure()
    mpl.rcParams['font.sans-serif'] = u'SimHei'
    mpl.rcParams['axes.unicode_minus'] = False
    #FPR就是横坐标,TPR就是纵坐标

    curve_recall = [ave_p_r_curve[kk][0] for kk in range(len(ave_p_r_curve))]
    curve_precision = [ave_p_r_curve[kk][1] for kk in range(len(ave_p_r_curve))]
    print('R-P:',P_R_curve_dict[key])
    print('x',curve_recall)
    print('y',curve_precision)
    # print('y',P_R_curve_dict[key][:])
    plt.plot(curve_recall,curve_precision, color='red', lw = 2, alpha = 0.7,label = 'mAP: %.4f'%(np.mean(MAP)))
    # plt.plot([0.1,0.5],[0.1,0.5], c = 'red', lw = 2, alpha = 0.7)
    # plt.plot((0, 1), (0, 1), c = '#808080', lw = 1, ls = '--', alpha = 0.7)
    #
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('Recall', fontsize=13)
    plt.ylabel('Precision', fontsize=13)
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
    plt.title(u'AVE P-R_multiLabel', fontsize=17)
    plt.show()

    fig.tight_layout()
    fig.tight_layout()
    #-------------------------------------------------------------------------------
    p_a = []
    r_a = []
    for key in P_R_chart_dict.keys():# 每一类别 P-R 表
        print('----------------->>> {}'.format(key))

        p_ = [P_R_chart_dict[key][i][1] for i in range(len(P_R_chart_dict[key]))]
        r_ = [P_R_chart_dict[key][i][0] for i in range(len(P_R_chart_dict[key]))]

        print('   label : {} - AP : {} , ARECALL : {}'.format(key,np.mean(p_),np.mean(r_)))
        p_a.append(np.mean(p_))
        r_a.append(np.mean(r_))
        # print(P_R_chart_dict[key])

    print('\n mAP : {}, mRECALL : {}'.format(np.mean(p_a),np.mean(r_a)))

    # for i in range(len(data)):
    #     scores,label = data[i]
    #     max_index = np.argmax(scores)
    #     confidence = score[max_index]
    #     print('label : gt {} ------ pre {}'.format(label,max_index))
    #
    #     if label not in P_R_dict.keys():
    #         P_R_dict[label] = []
    #     flag = 'n'
    #     if max_index == label:
    #         flag = 'p'
    #     P_R_dict[label].append()




    # P_R_dict
    # for thr in  thrs:
    #
    #     print('thr {:.3f}'.format(thr))
    #
    #     if thr not in P_R_dict.keys():
    #         P_R_dict[thr] = {}
    #
    #     for i in range(len(data)):
    #         score,label = data[i]
    #
    #         if label not in P_R_dict[thr].keys():
    #             P_R_dict[thr][label]={}
    #             P_R_dict[thr][label]['gt_num'] = 0 # 真实值
    #             P_R_dict[thr][label]['pre_right_num'] = 0 # 预测正确计数值
    #             P_R_dict[thr][label]['pre_num'] = 0 # 预测计数值
    #
    #         P_R_dict[thr][label]['gt_num'] += 1
    #
    #         #---------------------------------
    #         max_index = np.argmax(score)
    #
    #         if max_index not in P_R_dict[thr].keys():
    #             P_R_dict[thr][max_index]={}
    #             P_R_dict[thr][max_index]['gt_num'] = 0 # 真实值
    #             P_R_dict[thr][max_index]['pre_right_num'] = 0 # 预测正确计数值
    #             P_R_dict[thr][max_index]['pre_num'] = 0 # 预测计数值
    #
    #         P_R_dict[thr][label]['gt_num'] += 1
    #
    #         confidence = score[max_index]
    #
    #         # print(max_index,label,confidence,thr)
    #         if confidence >=thr :
    #             P_R_dict[thr][max_index]['pre_num'] += 1
    #             if max_index == label:
    #                 P_R_dict[thr][label]['pre_right_num'] += 1
    #
    #         #-------------------------------------
    #
    # # print('P_R_dict : \n',P_R_dict)
    # mp_list = []
    # mr_list = []
    # mp_list.append(1.0)
    # mr_list.append(0.0)
    # for thr in P_R_dict.keys():
    #     print(thr)
    #     mp = []
    #     mr = []
    #     for label in P_R_dict[thr].keys():
    #
    #         if P_R_dict[thr][label]['pre_num']>0:
    #             precision = float(P_R_dict[thr][label]['pre_right_num']) / float(P_R_dict[thr][label]['pre_num'])
    #         else:
    #             precision = 0.
    #         recall = float(P_R_dict[thr][label]['pre_right_num'])/float(P_R_dict[thr][label]['gt_num'])
    #
    #         # print('label-',label,' : ',P_R_dict[thr][label],'precision : {}, recall : {}'.format(precision,recall))
    #
    #         mp.append(precision)
    #         mr.append(recall)
    #     mp_list.append(np.mean(mp))
    #     mr_list.append(np.mean(mr))
    #
    # mp_list.append(0.0)
    # mr_list.append(1.0)
    #
    #
    #
    # mp = np.array(mp_list)
    # mr = np.array(mr_list)
    #
    # #绘图
    # mpl.rcParams['font.sans-serif'] = u'SimHei'
    # mpl.rcParams['axes.unicode_minus'] = False
    # #FPR就是横坐标,TPR就是纵坐标
    # plt.plot(mr, mp, c = 'r', lw = 2, alpha = 0.7)
    # plt.plot((0, 1), (0, 1), c = '#808080', lw = 1, ls = '--', alpha = 0.7)
    # #
    # # max_dst = 0.
    # # max_dst_x,max_dst_y = 0.,0.
    # # max_thr = 0.
    # for i in range(len(mp_list)):
    #
    #     plt.scatter(mr_list[i],mp_list[i],s=30, c='b', marker = 'o')
    # # plt.text(max_dst_x,max_dst_y, u'thr=%.5f'%(max_thr))
    # #
    # plt.xlim((-0.01, 1.02))
    # plt.ylim((-0.01, 1.02))
    # plt.xticks(np.arange(0, 1.1, 0.1))
    # plt.yticks(np.arange(0, 1.1, 0.1))
    # plt.xlabel('Recall', fontsize=13)
    # plt.ylabel('Precision', fontsize=13)
    # plt.grid(b=True, ls=':')
    # # plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
    # plt.title(u'P-R', fontsize=17)
    # print(ops.file.replace('.json','_pr.jpg'))
    # plt.savefig(ops.file.replace('.json','_pr.jpg'))
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=' ROC ')
    parser.add_argument('--file', type=str, default = './roc_metrics_2020-04-19_22-53-34.json',
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

    p_r(args)
