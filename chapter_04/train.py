#-*-coding:utf-8-*-
# date:2020-04-11
# Author: xiang li

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import  sys

# from tensorboardX import SummaryWriter
from utils.model_utils import *
from utils.common_utils import *
from data_iter.datasets import *
from loss.loss import FocalLoss
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

import cv2
import time
import json
from datetime import datetime

def trainer(ops):
    try:
        set_seed(ops.seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = ops.GPUS

        train_path =  ops.train_path
        num_classes = len(os.listdir(ops.train_path)) # 模型类别个数
        print('num_classes : ',num_classes)
        #---------------------------------------------------------------- 构建模型
        print('use model : %s'%(ops.model))

        if ops.model == 'resnet_18':
            model_=resnet18(pretrained = ops.pretrained)
        elif ops.model == 'resnet_34':
            model_=resnet34(pretrained = ops.pretrained)
        elif ops.model == 'resnet_50':
            model_=resnet50(pretrained = ops.pretrained)
        elif ops.model == 'resnet_101':
            model_=resnet101(pretrained = ops.pretrained)
        elif ops.model == 'resnet_152':
            model_=resnet152(pretrained = ops.pretrained)
        else:
            print('error no the struct model : {}'.format(ops.model))

        num_ftrs = model_.fc.in_features
        model_.fc = nn.Linear(num_ftrs, num_classes)

        use_cuda = torch.cuda.is_available()

        device = torch.device("cuda:0" if use_cuda else "cpu")
        model_ = model_.to(device)

        # print(model_)# 打印模型结构

        # Dataset
        dataset = LoadImagesAndLabels(path = ops.train_path,img_size=ops.img_size,flag_agu=ops.flag_agu,fix_res = ops.fix_res)
        print('len train datasets : %s'%(dataset.__len__()))
        # Dataloader
        dataloader = DataLoader(dataset,
                                batch_size=ops.batch_size,
                                num_workers=ops.num_workers,
                                shuffle=True,
                                pin_memory=False,
                                drop_last = True)
        # 优化器设计
        # optimizer_Adam = torch.optim.Adam(model_.parameters(), lr=init_lr, betas=(0.9, 0.99),weight_decay=1e-6)
        optimizer_SGD = optim.SGD(model_.parameters(), lr=ops.init_lr, momentum=0.9, weight_decay=ops.weight_decay)# 优化器初始化
        optimizer = optimizer_SGD
        # 加载 finetune 模型
        if os.access(ops.fintune_model,os.F_OK):# checkpoint
            chkpt = torch.load(ops.fintune_model, map_location=device)
            model_.load_state_dict(chkpt)
            print('load fintune model : {}'.format(ops.fintune_model))

        print('/**********************************************/')
        # 损失函数
        if 'focalLoss' == ops.loss_define:
            criterion = FocalLoss(num_class = num_classes)
        else:
            criterion = nn.CrossEntropyLoss()#CrossEntropyLoss() 是 softmax 和 负对数损失的结合

        step = 0
        idx = 0

        # 变量初始化
        best_loss = np.inf
        loss_mean = 0. # 损失均值
        loss_idx = 0. # 损失计算计数器
        flag_change_lr_cnt = 0 # 学习率更新计数器
        init_lr = ops.init_lr # 学习率

        for epoch in range(0, ops.epochs):
            print('\nepoch %d ------>>>'%epoch)
            model_.train()
            # 学习率更新策略
            if loss_mean!=0.:
                if best_loss > (loss_mean/loss_idx):
                    flag_change_lr_cnt = 0
                    best_loss = (loss_mean/loss_idx)
                else:
                    flag_change_lr_cnt += 1

                    if flag_change_lr_cnt >=2:
                        init_lr = init_lr*ops.lr_decay
                        set_learning_rate(optimizer, init_lr)
                        flag_change_lr_cnt = 0

            for i, (imgs_, labels_) in enumerate(dataloader):

                if use_cuda:
                    imgs_ = imgs_.cuda()  # pytorch 的 数据输入格式 ： (batch, channel, height, width)
                    labels_ = labels_.cuda()

                output = model_(imgs_.float())

                loss = criterion(output, labels_)
                loss_mean += loss.item()
                loss_idx += 1.
                if i%10 == 0:
                    acc = get_acc(output, labels_)
                    loc_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    print('   %s - %s - epoch [%s/%s] (%s/%s): '%(loc_time,ops.model,epoch,ops.epochs,i,int(dataset.__len__()/ops.batch_size)),\
                    ' loss : %.6f - %.6f'%(loss_mean/loss_idx,loss.item()),\
                    ' acc : %.4f'%acc,' lr : %.5f'%init_lr,' bs : ',ops.batch_size,\
                    ' img_size : %s x %s'%(ops.img_size[0],ops.img_size[1]),' best_loss : %.4f'%best_loss)
                    # time.sleep(1)
                    # writer.add_scalar('data/loss', loss, step)
                    # writer.add_scalars('data/scalar_group', {'acc':acc,'lr':init_lr,'baseline':0.}, step)

                # 计算梯度
                loss.backward()
                # 优化器对模型参数更新
                optimizer.step()
                # 优化器梯度清零
                optimizer.zero_grad()
                step += 1

                # 一个 epoch 保存连词最新的 模型
                if i%(int(dataset.__len__()/ops.batch_size/2-1)) == 0 and i > 0:
                    torch.save(model_.state_dict(), ops.model_exp + 'latest.pth')
            # 每一个 epoch 进行模型保存
            torch.save(model_.state_dict(), ops.model_exp + 'model_epoch-{}.pth'.format(epoch))
    except Exception as e:
        print('Exception : ',e) # 打印异常
        print('Exception  file : ', e.__traceback__.tb_frame.f_globals['__file__'])# 发生异常所在的文件
        print('Exception  line : ', e.__traceback__.tb_lineno)# 发生异常所在的行数

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=' Project Classification Train')
    parser.add_argument('--seed', type=int, default = 999,
        help = 'seed') # 设置随机种子
    parser.add_argument('--model_exp', type=str, default = './model_exp',
        help = 'model_exp') # 模型输出文件夹
    parser.add_argument('--model', type=str, default = 'resnet_101',
        help = 'model : resnet_18,resnet_34,resnet_50,resnet_101,resnet_152') # 模型类型
    parser.add_argument('--GPUS', type=str, default = '0',
        help = 'GPUS') # GPU选择
    parser.add_argument('--train_path', type=str, default = './datasets/train_datasets/',
        help = 'train_path') # 训练集路径
    parser.add_argument('--pretrained', type=bool, default = True,
        help = 'imageNet_Pretrain') # 初始化学习率
    parser.add_argument('--fintune_model', type=str, default = 'None',
        help = 'fintune_model') # fintune model
    parser.add_argument('--loss_define', type=str, default = True,
        help = 'define_loss') # 损失函数定义
    parser.add_argument('--init_lr', type=float, default = 1e-3,
        help = 'init_learningRate') # 初始化学习率
    parser.add_argument('--lr_decay', type=float, default = 0.9,
        help = 'learningRate_decay') # 学习率权重衰减率
    parser.add_argument('--weight_decay', type=float, default = 1e-8,
        help = 'weight_decay') # 优化器正则损失权重
    parser.add_argument('--batch_size', type=int, default = 32,
        help = 'batch_size') # 训练每批次图像数量
    parser.add_argument('--epochs', type=int, default = 1000,
        help = 'epochs') # 训练周期
    parser.add_argument('--num_workers', type=int, default = 6,
        help = 'num_workers') # 训练数据生成器线程数
    parser.add_argument('--img_size', type=tuple , default = (224,224),
        help = 'img_size') # 输入模型图片尺寸
    parser.add_argument('--flag_agu', type=bool , default = True,
        help = 'data_augmentation') # 训练数据生成器是否进行数据扩增
    parser.add_argument('--fix_res', type=bool , default = True,
        help = 'fix_resolution') # 输入模型样本图片是否保证图像分辨率的长宽比
    parser.add_argument('--clear_model_exp', type=bool, default = True,
        help = 'clear_model_exp') # 模型输出文件夹是否进行清除

    print('\n/******************* {} ******************/\n'.format(parser.description))
    #--------------------------------------------------------------------------
    args = parser.parse_args()# 解析添加参数
    #--------------------------------------------------------------------------
    mkdir_(args.model_exp, flag_rm=args.clear_model_exp)
    loc_time = time.localtime()
    args.model_exp = args.model_exp + '/' + time.strftime("%Y-%m-%d_%H-%M-%S", loc_time)+'/'
    mkdir_(args.model_exp, flag_rm=args.clear_model_exp)

    print('----------------------------------')

    unparsed = vars(args) # parse_args()方法的返回值为namespace，用vars()内建函数化为字典
    for key in unparsed.keys():
        print('{} : {}'.format(key,unparsed[key]))

    unparsed['time'] = time.strftime("%Y-%m-%d %H:%M:%S", loc_time)

    fs = open(args.model_exp+'train_ops.json',"w",encoding='utf-8')
    json.dump(unparsed,fs,ensure_ascii=False,indent = 1)
    fs.close()

    trainer(ops = args)# 模型训练

    print('well done : {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
