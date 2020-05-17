#-*-coding:utf-8-*-
# date:2019-08
# Author: xiang.li
# function: training
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from yolov3 import Yolov3, Yolov3Tiny
from utils.parse_config import parse_data_cfg
from utils.torch_utils import select_device
import torch
from torch.utils.data import DataLoader
from utils.datasets import LoadImagesAndLabels
from utils.utils import *
import numpy as np
import test

def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_model(model, pretrained_state_dict):

    print('pretrained_state_dict : \n',pretrained_state_dict.keys())
    pretrained_state_dict = pretrained_state_dict['model']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_state_dict.items() if
                       k in model_dict and model_dict[k].size() == pretrained_state_dict[k].size()}
    model.load_state_dict(pretrained_dict, strict=False)
    if len(pretrained_dict) == 0:
        print("[INFO] No params were loaded ...")
    else:
        for k, v in pretrained_state_dict.items():
            if k in pretrained_dict:
                print("==>> Load {} {}".format(k, v.size()))
            else:
                print("[INFO] Skip {} {}".format(k, v.size()))
    return model

def train(data_cfg ='cfg/voc.data',
    accumulate = 1):
    device = select_device()
    # Config
    get_data_cfg = parse_data_cfg(data_cfg)#返回训练配置参数，类型：字典

    num_workers = int(get_data_cfg['num_workers'])
    cfg_model = get_data_cfg['cfg_model']
    train_path = get_data_cfg['train']
    valid_ptah = get_data_cfg['valid']
    num_classes = int(get_data_cfg['classes'])
    finetune_model = get_data_cfg['finetune_model']
    batch_size = int(get_data_cfg['batch_size'])
    img_size = int(get_data_cfg['img_size'])
    multi_scale = get_data_cfg['multi_scale']
    epochs = int(get_data_cfg['epochs'])
    lr_step = str(get_data_cfg['lr_step'])

    if multi_scale == 'True':
        multi_scale = True
    else:
        multi_scale = False

    print('data_cfg            : ',data_cfg)
    print('voc.data config len : ',len(get_data_cfg))
    print('num_workers      : ',num_workers)
    print('model            : ',cfg_model)
    print('finetune_model   : ',finetune_model)
    print('train_path       : ',train_path)
    print('valid_ptah       : ',valid_ptah)
    print('num_classes      : ',num_classes)
    print('batch_size       : ',batch_size)
    print('img_size         : ',img_size)
    print('multi_scale      : ',multi_scale)
    print('lr_step          : ',lr_step)
    # load model
    if "tiny" in cfg_model:
        a_scalse = 416./img_size
        anchors=[(10, 14), (23, 27), (37, 58), (81, 82), (135, 169), (344, 319)]
        anchors_new = [ (int(anchors[j][0]/a_scalse),int(anchors[j][1]/a_scalse)) for j in range(len(anchors)) ]
        print('old anchors : ',anchors)
        model = Yolov3Tiny(num_classes,anchors = anchors_new)
        weights = './weights-yolov3-tiny/'
    else:
        a_scalse = 416./img_size
        anchors=[(10,13), (16,30), (33,23), (30,61), (62,45), (59,119), (116,90), (156,198), (373,326)]
        anchors_new = [ (int(anchors[j][0]/a_scalse),int(anchors[j][1]/a_scalse)) for j in range(len(anchors)) ]
        model = Yolov3(num_classes,anchors = anchors_new)
        weights = './weights-yolov3/'
    # make dir save model document
    if not os.path.exists(weights):
        os.mkdir(weights)


    latest = weights + 'latest.pt'
    best = weights + 'best.pt'
    # Optimizer
    lr0 = 0.001  # initial learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr0, momentum=0.9, weight_decay=0.0005)

    start_epoch = 0
    model = model.to(device)

    print (finetune_model)

    if os.access(finetune_model,os.F_OK):
        print('\n/************************** load_model *************************/')
        print (finetune_model)
        load_model(model, torch.load(finetune_model))
    else:
        print('finetune_model not exist !')

    milestones=[int(i) for i in lr_step.split(",")]
    print('milestones : ',milestones)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(i) for i in lr_step.split(",")], gamma=0.1,
                                                     last_epoch=start_epoch - 1)

    # Initialize distributed training
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend=opt.backend, init_method=opt.dist_url, world_size=opt.world_size, rank=opt.rank)
        model = torch.nn.parallel.DistributedDataParallel(model)

    # Dataset
    print('multi_scale : ',multi_scale)
    dataset = LoadImagesAndLabels(train_path, batch_size=batch_size, img_size=img_size, augment=True, multi_scale=multi_scale)
    print('--------------->>> imge num : ',dataset.__len__())
    # Dataloader
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=True,
                            pin_memory=False,
                            drop_last = False,
                            collate_fn=dataset.collate_fn)

    # Start training
    t = time.time()
    model_info(model)
    nB = len(dataloader)
    n_burnin = min(round(nB / 5 + 1), 1000)  # burn-in batches

    best_loss = float('inf')
    test_loss = float('inf')
    start_epoch = 0
    for epoch in range(start_epoch, epochs):

        print('')
        model.train()

        scheduler.step()

        mloss = defaultdict(float)  # mean loss
        for i, (imgs, targets, img_path_, _) in enumerate(dataloader):
            multi_size = imgs.size()
            imgs = imgs.to(device)
            targets = targets.to(device)

            nt = len(targets)
            if nt == 0:  # if no targets continue
                continue

            # SGD burn-in
            if epoch == 0 and i <= n_burnin:
                lr = lr0 * (i / n_burnin) ** 4
                for x in optimizer.param_groups:
                    x['lr'] = lr

            # Run model
            pred = model(imgs)

            # Build targets
            target_list = build_targets(model, targets)

            # Compute loss
            loss, loss_dict = compute_loss(pred, target_list)

            # Compute gradient
            loss.backward()

            # Accumulate gradient for x batches before optimizing
            if (i + 1) % accumulate == 0 or (i + 1) == nB:
                optimizer.step()
                optimizer.zero_grad()

            # Running epoch-means of tracked metrics
            for key, val in loss_dict.items():
                mloss[key] = (mloss[key] * i + val) / (i + 1)

            print('Epoch {:3d}/{:3d}, Batch {:6d}/{:6d}, Img_size {}x{}, nTargets {}, lr {:.6f}, loss: xy {:.2f}, wh {:.2f}, '
                  'conf {:.2f}, cls {:.2f}, total {:.2f}, time {:.3f}s'.format(epoch, epochs - 1, i, nB - 1, multi_size[2], multi_size[3]
                   , nt, scheduler.get_lr()[0], mloss['xy'], mloss['wh'], mloss['conf'], mloss['cls'], mloss['total'], time.time() - t),
                   end = '\r')

            s = ('%8s%12s' + '%10.3g' * 7) % ('%g/%g' % (epoch, epochs - 1), '%g/%g' % (i, nB - 1), mloss['xy'],
                mloss['wh'], mloss['conf'], mloss['cls'], mloss['total'], nt, time.time() - t)
            t = time.time()
            if i%300 == 0 and i>0:
                # Create checkpoint
                chkpt = {'epoch': epoch,
                         'best_loss': best_loss,
                         'model': model.module.state_dict() if type(
                             model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                         'optimizer': optimizer.state_dict()}

                # Save latest checkpoint
                torch.save(chkpt, latest)

        if epoch%20 == 0 and epoch >0:
            # Calculate mAP
            print('\n')
            with torch.no_grad():
                print("-------"*5 + "testing" + "-------"*5)
                results = test.test(cfg_model, data_cfg, batch_size=batch_size, img_size=img_size, model=model)
            # Update best loss
            test_loss = results[4]
            if test_loss < best_loss:
                best_loss = test_loss

        if True:
            # Create checkpoint
            chkpt = {'epoch': epoch,
                     'best_loss': best_loss,
                     'model': model.module.state_dict() if type(
                         model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                     'optimizer': optimizer.state_dict()}

            # Save latest checkpoint
            torch.save(chkpt, latest)

            # Save best checkpoint
            if best_loss == test_loss and epoch%5 == 0:
                torch.save(chkpt, best)

            # Save backup every 10 epochs (optional)
            if epoch > 0 and epoch % 5 == 0:
                torch.save(chkpt, weights + 'Detect%g.pt' % epoch)

            # Delete checkpoint
            del chkpt

#-------------------------------------------------------------------------------
if __name__ == '__main__':

    train(data_cfg='cfg/voc_faces.data')

    print('well done ~ ')
