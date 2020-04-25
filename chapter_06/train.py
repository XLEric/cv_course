#-*-coding:utf-8-*-
# date:2020-04-24
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

from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from loss.loss import *
import cv2
import time
import json
from datetime import datetime
#
# def tester(ops,epoch,model,criterion,
#     train_split,train_split_label,val_split,val_split_label,
#     use_cuda):
#     #
#     print('\n------------------------->>> tester traival loss')
#
#     loss_train = []
#     loss_val = []
#     with torch.no_grad():
#         # train loss
#         for i in range(len(train_split)):
#             file = train_split[i]
#             label = train_split_label[i]
#
#             img = cv2.imread(file)
#             # 输入图片预处理
#             if ops.fix_res:
#                 img_ = letterbox(img,size_=ops.img_size[0],mean_rgb = (128,128,128))
#             else:
#                 img_ = cv2.resize(img, (ops.img_size[1],ops.img_size[0]), interpolation = cv2.INTER_CUBIC)
#
#             img_ = img_.astype(np.float32)
#             img_ = (img_-128.)/256.
#
#             img_ = img_.transpose(2, 0, 1)
#             img_ = torch.from_numpy(img_)
#             img_ = img_.unsqueeze_(0)
#
#             label_ = np.array(label)
#             label_ = torch.from_numpy(label_).float()
#
#             if use_cuda:
#                 img_ = img_.cuda()  # (bs, 3, h, w)
#                 labels_ = label_.cuda()  # (bs, 3, h, w)
#
#
#             output = model(img_.float())
#
#             loss = criterion(output, labels_)
#             loss_train.append(loss.item())
#         # val loss
#         for i in range(len(val_split)):
#             file = val_split[i]
#             label = val_split_label[i]
#
#             img = cv2.imread(file)
#             # 输入图片预处理
#             if ops.fix_res:
#                 img_ = letterbox(img,size_=ops.img_size[0],mean_rgb = (128,128,128))
#             else:
#                 img_ = cv2.resize(img, (ops.img_size[1],ops.img_size[0]), interpolation = cv2.INTER_CUBIC)
#
#             img_ = img_.astype(np.float32)
#             img_ = (img_-128.)/256.
#
#             img_ = img_.transpose(2, 0, 1)
#             img_ = torch.from_numpy(img_)
#             img_ = img_.unsqueeze_(0)
#
#             label_ = np.array(label)
#             label_ = torch.from_numpy(label_).float()
#
#             if use_cuda:
#                 img_ = img_.cuda()  # (bs, 3, h, w)
#                 labels_ = label_.cuda()  # (bs, 3, h, w)
#
#
#             output = model(img_.float())
#
#             loss = criterion(output, labels_)
#             loss_val.append(loss.item())
#
#     print('loss_train : {}, loss_val : {} '.format(np.mean(loss_train),np.mean(loss_val)))
#
#     return np.mean(loss_train),np.mean(loss_val)


def trainer(ops,f_log):
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = ops.GPUS

        if ops.log_flag:
            sys.stdout = f_log

        set_seed(ops.seed)
        #---------------------------------------------------------------- 构建模型
        print('use model : %s'%(ops.model))

        if ops.model == 'resnet_18':
            model_=resnet18(pretrained = ops.pretrained, num_classes=ops.num_classes, img_size=ops.img_size[0],dropout_factor=ops.dropout)
        elif ops.model == 'resnet_34':
            model_=resnet34(pretrained = ops.pretrained, num_classes=ops.num_classes, img_size=ops.img_size[0],dropout_factor=ops.dropout)
        elif ops.model == 'resnet_50':
            model_=resnet50(pretrained = ops.pretrained, num_classes=ops.num_classes, img_size=ops.img_size[0],dropout_factor=ops.dropout)
        elif ops.model == 'resnet_101':
            model_=resnet101(pretrained = ops.pretrained, num_classes=ops.num_classes, img_size=ops.img_size[0],dropout_factor=ops.dropout)
        elif ops.model == 'resnet_152':
            model_=resnet152(pretrained = ops.pretrained, num_classes=ops.num_classes, img_size=ops.img_size[0],dropout_factor=ops.dropout)
        else:
            print('error no the struct model : {}'.format(ops.model))

        print('model_.fc : {}'.format(model_.fc))


        use_cuda = torch.cuda.is_available()

        device = torch.device("cuda:0" if use_cuda else "cpu")
        model_ = model_.to(device)

        # print(model_)# 打印模型结构
        # Dataset
        val_split = []
        dataset = LoadImagesAndLabels(ops= ops,img_size=ops.img_size,flag_agu=ops.flag_agu,fix_res = ops.fix_res,val_split = val_split)
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
        if ops.loss_define != 'wing_loss':
            criterion = nn.MSELoss(reduce=True, reduction='mean')

        step = 0
        idx = 0

        # 变量初始化
        best_loss = np.inf
        loss_mean = 0. # 损失均值
        loss_idx = 0. # 损失计算计数器
        flag_change_lr_cnt = 0 # 学习率更新计数器
        init_lr = ops.init_lr # 学习率

        epochs_loss_dict = {}

        for epoch in range(0, ops.epochs):
            if ops.log_flag:
                sys.stdout = f_log
            print('\nepoch %d ------>>>'%epoch)
            model_.train()
            # 学习率更新策略
            if loss_mean!=0.:
                if best_loss > (loss_mean/loss_idx):
                    flag_change_lr_cnt = 0
                    best_loss = (loss_mean/loss_idx)
                else:
                    flag_change_lr_cnt += 1

                    if flag_change_lr_cnt > 5:
                        init_lr = init_lr*ops.lr_decay
                        set_learning_rate(optimizer, init_lr)
                        flag_change_lr_cnt = 0

            loss_mean = 0. # 损失均值
            loss_idx = 0. # 损失计算计数器

            for i, (imgs_, pts_) in enumerate(dataloader):
                # print('imgs_, pts_',imgs_.size(), pts_.size())
                if use_cuda:
                    imgs_ = imgs_.cuda()  # pytorch 的 数据输入格式 ： (batch, channel, height, width)
                    pts_ = pts_.cuda()

                output = model_(imgs_.float())
                if ops.loss_define == 'wing_loss':
                    loss = got_total_wing_loss(output, pts_.float())
                else:
                    loss = criterion(output, pts_.float())
                loss_mean += loss.item()
                loss_idx += 1.
                if i%10 == 0:
                    loc_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    print('  %s - %s - epoch [%s/%s] (%s/%s):'%(loc_time,ops.model,epoch,ops.epochs,i,int(dataset.__len__()/ops.batch_size)),\
                    'loss : %.6f - %.6f'%(loss_mean/loss_idx,loss.item()),\
                    ' lr : %.5f'%init_lr,' bs :',ops.batch_size,\
                    ' img_size: %s x %s'%(ops.img_size[0],ops.img_size[1]),' best_loss: %.4f'%best_loss)
                    time.sleep(5)
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
                # if i%(int(dataset.__len__()/ops.batch_size/2-1)) == 0 and i > 0:
                #     torch.save(model_.state_dict(), ops.model_exp + 'latest.pth')
            # 每一个 epoch 进行模型保存
            torch.save(model_.state_dict(), ops.model_exp + 'model_epoch-{}.pth'.format(epoch))

            if len(val_split) > 0 and (epoch%ops.test_interval==0): # test

                model_.eval()
                loss_train,loss_val = tester(ops,epoch,model_,criterion,
                        train_split,train_split_label,val_split,val_split_label,
                        use_cuda)

                epochs_loss_dict['epoch_'+str(epoch)] = {}

                epochs_loss_dict['epoch_'+str(epoch)]['loss_train'] = loss_train
                epochs_loss_dict['epoch_'+str(epoch)]['loss_val'] = loss_val

                f_loss = open(ops.model_exp + 'loss_epoch_trainval.json',"w",encoding='utf-8')
                json.dump(epochs_loss_dict,f_loss,ensure_ascii=False,indent = 1,cls = JSON_Encoder)
                f_loss.close()

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
    parser.add_argument('--model', type=str, default = 'resnet_50',
        help = 'model : resnet_18,resnet_34,resnet_50,resnet_101,resnet_152') # 模型类型
    parser.add_argument('--num_classes', type=int , default = 196,
        help = 'num_classes') #  landmarks 个数*2
    parser.add_argument('--GPUS', type=str, default = '6',
        help = 'GPUS') # GPU选择

    parser.add_argument('--images_path', type=str, default = './datasets/WFLW_images/',
        help = 'images_path') # 图片路径

    parser.add_argument('--train_list', type=str,
        default = './datasets/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt',
        help = 'annotations_train_list')# 训练集标注信息
    parser.add_argument('--test_list', type=str,
        default = './datasets/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt',
        help = 'annotations_test_list')# 测试集标注信息

    parser.add_argument('--val_factor', type=float, default = 0.1,
        help = 'val_factor') # 从训练集中分离验证集对应的比例
    parser.add_argument('--test_interval', type=int, default = 1,
        help = 'test_interval') # 训练集和测试集 计算 loss 间隔
    parser.add_argument('--pretrained', type=bool, default = True,
        help = 'imageNet_Pretrain') # 初始化学习率
    parser.add_argument('--fintune_model', type=str, default = './model_exp/2020-04-25_14-48-25/model_epoch-1.pth',
        help = 'fintune_model') # fintune model
    parser.add_argument('--loss_define', type=str, default = 'wing_loss',
        help = 'define_loss') # 损失函数定义
    parser.add_argument('--init_lr', type=float, default = 1e-3,
        help = 'init_learningRate') # 初始化学习率
    parser.add_argument('--lr_decay', type=float, default = 0.96,
        help = 'learningRate_decay') # 学习率权重衰减率
    parser.add_argument('--weight_decay', type=float, default = 1e-8,
        help = 'weight_decay') # 优化器正则损失权重
    parser.add_argument('--batch_size', type=int, default = 16,
        help = 'batch_size') # 训练每批次图像数量
    parser.add_argument('--dropout', type=float, default = 0.5,
        help = 'dropout') # dropout
    parser.add_argument('--epochs', type=int, default = 1000,
        help = 'epochs') # 训练周期
    parser.add_argument('--num_workers', type=int, default = 6,
        help = 'num_workers') # 训练数据生成器线程数
    parser.add_argument('--img_size', type=tuple , default = (256,256),
        help = 'img_size') # 输入模型图片尺寸
    parser.add_argument('--flag_agu', type=bool , default = True,
        help = 'data_augmentation') # 训练数据生成器是否进行数据扩增
    parser.add_argument('--fix_res', type=bool , default = False,
        help = 'fix_resolution') # 输入模型样本图片是否保证图像分辨率的长宽比
    parser.add_argument('--clear_model_exp', type=bool, default = False,
        help = 'clear_model_exp') # 模型输出文件夹是否进行清除
    parser.add_argument('--log_flag', type=bool, default = False,
        help = 'log flag') # 是否保存训练 log

    #--------------------------------------------------------------------------
    args = parser.parse_args()# 解析添加参数
    #--------------------------------------------------------------------------
    mkdir_(args.model_exp, flag_rm=args.clear_model_exp)
    loc_time = time.localtime()
    args.model_exp = args.model_exp + '/' + time.strftime("%Y-%m-%d_%H-%M-%S", loc_time)+'/'
    mkdir_(args.model_exp, flag_rm=args.clear_model_exp)

    f_log = None
    if args.log_flag:
        f_log = open(args.model_exp+'/train_{}.log'.format(time.strftime("%Y-%m-%d_%H-%M-%S",loc_time)), 'a+')
        sys.stdout = f_log

    print('---------------------------------- log : {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", loc_time)))
    print('\n/******************* {} ******************/\n'.format(parser.description))

    unparsed = vars(args) # parse_args()方法的返回值为namespace，用vars()内建函数化为字典
    for key in unparsed.keys():
        print('{} : {}'.format(key,unparsed[key]))

    unparsed['time'] = time.strftime("%Y-%m-%d %H:%M:%S", loc_time)

    fs = open(args.model_exp+'train_ops.json',"w",encoding='utf-8')
    json.dump(unparsed,fs,ensure_ascii=False,indent = 1)
    fs.close()

    trainer(ops = args,f_log = f_log)# 模型训练

    if args.log_flag:
        sys.stdout = f_log
    print('well done : {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
