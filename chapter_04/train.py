#-*-coding:utf-8-*-
# date:2020-01-28
# Author: xiang li

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
# from tensorboardX import SummaryWriter

from data_iter.datasets import *
from loss.loss import LabelSmoothing,FocalLoss
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

import cv2
import numpy as np

import random
import time
from datetime import datetime

def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / float(total)

def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def init_seed(seed = 666):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        # cudnn.benchmark = False
        # cudnn.enabled = False

def trainer():
    pass

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "7"
    init_seed()

    save_model_dir= './model_s_dir/'
    model_dir = save_model_dir
    if not os.path.exists(save_model_dir):
        os.mkdir(save_model_dir)


    train_path =  './datasets/train_datasets/'
    print('datasets label : %s'%(os.listdir(train_path)))
    output_node = len(os.listdir(train_path))
    print('output_node : %s'%(output_node))
    model_name = "resnet50"
    print('use model : %s'%(model_name))
    # Number of classes in the dataset
    num_classes = output_node # 模型类别个数

    feature_extract = False

    name='resnet50'
    model_=resnet50(pretrained = False,num_classes=num_classes)
    num_ftrs = model_.fc.in_features
    model_.fc = nn.Linear(num_ftrs, num_classes)

    print('num_ftrs : ',num_ftrs,' num_classes : ',num_classes)

    # writer = SummaryWriter(logdir='./logs_train', comment=model_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ = model_.to(device)

    print(model_)

    init_lr = 1e-3

    batch_size = 64
    start_epoch = 0
    epochs = 1000
    num_workers = 6
    img_size = (224,224)
    lr_decay_step = 1
    flag_agu = True
    fix_res = True
    backward_decay_step = 1
    print('image size    :',img_size)
    print('batch_size    : ',batch_size)
    print('num_workers   : ',num_workers)
    print('init_lr       : ',init_lr)
    print('epochs        : ',epochs)
    # Dataset
    dataset = LoadImagesAndLabels(path = train_path,img_size=img_size,flag_agu=flag_agu,fix_res = fix_res)
    print('len train datasets : %s'%(dataset.__len__()))
    # # Dataloader
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=True,
                            pin_memory=False,
                            drop_last = True)

    # optimizer_Adam = torch.optim.Adam(model_.parameters(), lr=init_lr, betas=(0.9, 0.99),weight_decay=1e-6)
    optimizer_SGD = optim.SGD(model_.parameters(), lr=init_lr, momentum=0.9, weight_decay=1e-8)# 优化器初始化
    optimizer = optimizer_SGD

    model_path = model_dir+'model_epoch-3.pth'
    if os.access(model_path,os.F_OK):# checkpoint
        chkpt = torch.load(model_path, map_location=device)
        model_.load_state_dict(chkpt)
        print('load model : ',model_path)


    print('/**********************************************/')

    loss_define = 'focal_loss'

    if 'focal_loss' == loss_define:
        criterion = FocalLoss(num_class = num_classes)
    else:
        criterion = nn.CrossEntropyLoss()#CrossEntropyLoss() 是 softmax 和 负对数损失的结合

    step = 0
    idx = 0
    use_cuda = torch.cuda.is_available()
    test_moment = 1

    best_loss = np.inf
    loss_mean = 0.
    loss_idx = 0.
    flag_change_lr_cnt = 0

    for epoch in range(start_epoch, epochs):
        print('\nepoch %d ------>>>'%epoch)
        model_.train()

        if loss_mean!=0.:
            if best_loss > (loss_mean/loss_idx):
                flag_change_lr_cnt = 0
                best_loss = (loss_mean/loss_idx)
            else:
                flag_change_lr_cnt += 1

                if flag_change_lr_cnt >=2:
                    init_lr = init_lr*0.9
                    set_learning_rate(optimizer, init_lr)
                    flag_change_lr_cnt = 0

        for i, (imgs_, labels_) in enumerate(dataloader):

            if use_cuda:
                imgs_ = imgs_.cuda()  # (bs, 3, h, w)
                labels_ = labels_.cuda()


            output = model_(imgs_.float())


            loss = criterion(output, labels_)
            loss_mean += loss.item()
            loss_idx += 1.
            if i%10 == 0:
                acc = get_acc(output, labels_)
                print('       %s - epoch [%s/%s] (%s/%s): '%(model_name,epoch,epochs,i,int(dataset.__len__()/batch_size)),' loss : %.6f - %.6f'%(loss_mean/loss_idx,loss.item()),' acc : %.4f'%acc,' lr : %.5f'%init_lr,' bs : ',batch_size,\
                ' img_size : %s x %s'%(img_size[0],img_size[1]),' best_loss : %.4f'%best_loss)
                # time.sleep(1)
                # writer.add_scalar('data/loss', loss, step)
                # writer.add_scalars('data/scalar_group', {'acc':acc,'lr':init_lr,'baseline':0.}, step)

            # Compute gradient
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            step += 1


            if i%(int(dataset.__len__()/batch_size/2-1)) == 0 and i > 0:
                torch.save(model_.state_dict(), save_model_dir + 'latest.pth')

        torch.save(model_.state_dict(), save_model_dir + 'model_epoch-{}.pth'.format(epoch))

    # writer.close()

    print('well done ')
