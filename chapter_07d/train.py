#-*-coding:utf-8-*-
# date:2019-09-20
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.append('./')
import numpy as np
from detect_iterator.detect_data_iterator import *
from tools import *
import xml.etree.cElementTree as ET
from models.RFB_Net_vgg import build_net
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Variable
from layers.modules import MultiBoxLoss
from layers.functions import PriorBox
import time

#RFB CONFIGS
VOC_300 = {
    'num_classes': 2,
    'img_dim' : 300,
    'rgb_means' : (104, 117, 123),
    'p': 0.6,
    'feature_maps' : [38, 19, 10, 5, 3, 1],
    'min_dim' : 300,
    'steps' : [8, 16, 32, 64, 100, 300],
    'min_sizes' : [30, 60, 111, 162, 213, 264],
    'max_sizes' : [60, 111, 162, 213, 264, 315],
    'aspect_ratios' : [[2,3], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance' : [0.1, 0.2],
    'clip' : True,
}

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 2, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 4, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# labels_name = ['person', 'aeroplane', 'tvmonitor', 'train', 'boat', 'dog',\
# 'chair', 'bird', 'bicycle', 'bottle', 'sheep', 'diningtable', 'horse', 'motorbike',\
#  'sofa', 'cow', 'car', 'cat', 'bus', 'pottedplant']

labels_name = ['face']
if __name__ == "__main__":

    save_model_dir= './model_detect_dir/'# 模型保存路径
    model_dir = save_model_dir
    if not os.path.exists(save_model_dir):
        os.mkdir(save_model_dir)

    flag_restart = True
    path_pretrain_basenet_ ='./weights/vgg16_reducedfc.pth'

    train_path = './WIDER_FACE_VOC/'

    img_size = (300,300)# h,w is same
    if img_size[0] != img_size[1]:
        print('img size define error ,(h,w) must be same (300 or 512)')
    img_dim = img_size[0]
    num_workers = 4
    rgb_means = (104, 117, 123)
    p = (0.6)
    num_classes = 2 # voc
    init_lr = 4e-3
    gamma = 0.1

    batch_size = 8
    momentum = 0.9
    weight_decay = 0.0005
    start_epoch = 0
    epochs = 300
    lr_decay_step = 1
    start_epoch = 0
    augment = True
    is_train = True
    print('------------>>> step 1')
    dataset = LoadImagesAndLabels(path = train_path,img_size=img_size,rgb_means= rgb_means,is_train=is_train, augment=augment)
    print('------------>>> step 2')
    # Dataloader
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=True,
                            pin_memory=False,
                            drop_last = False,
                            collate_fn=detection_collate# 图片和 label ，对应 上 dim = 0 batch
                            )

    #------------------------------------------------
    device = select_device()
    model_ = build_net('train', img_dim, num_classes)

    if os.access(save_model_dir+'RFB_vgg_VOC_latest.pt',os.F_OK) and flag_restart == False:# checkpoint
        my_model_path = model_dir+'RFB_vgg_VOC_latest.pt'
        chkpt = torch.load(my_model_path, map_location=device)
        print('device:',device)
        model_.load_state_dict(chkpt['model'])
        start_epoch = chkpt['epoch']
        # if start_epoch>0:
        #     init_lr = chkpt['init_lr']/0.9
        # else:
        init_lr = chkpt['init_lr']
        # init_lr = 0.01
        print('load retrain model : ',save_model_dir+'RFB_vgg_VOC_latest.pt')
    else:
        if flag_restart:
            print('Loading base network... ',path_pretrain_basenet_)
            base_weights = torch.load(path_pretrain_basenet_)
            model_.base.load_state_dict(base_weights)

            def xavier(param):
                init.xavier_uniform(param)

            def weights_init(m):
                for key in m.state_dict():
                    if key.split('.')[-1] == 'weight':
                        if 'conv' in key:
                            init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                        if 'bn' in key:
                            m.state_dict()[key][...] = 1
                    elif key.split('.')[-1] == 'bias':
                        m.state_dict()[key][...] = 0
            model_.extras.apply(weights_init)
            model_.loc.apply(weights_init)
            model_.conf.apply(weights_init)
            model_.Norm.apply(weights_init)

    optimizer = optim.SGD(model_.parameters(), lr=init_lr,momentum=momentum, weight_decay=weight_decay)
    # model_ = model_.to(device)
    model_.cuda()
    model_.train()
    use_cuda = torch.cuda.is_available()
    model_.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0

    criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False)

    priorbox = PriorBox(VOC_300,debug_ = False)

    with torch.no_grad():
        priors = priorbox.forward()
        if use_cuda:
            priors = priors.cuda()
            print('priors.size(0) : ',priors.size(0))
    #------------------------------------------------

    epochs = 1000

    # (104,117,123)
    best_loss = np.inf
    loss_mean = np.inf
    loss_idx = 0.

    img_idx = 0
    c_cnt = 0
    for epoch in range(start_epoch, epochs):
        model_.train()

        lr = init_lr
        if epoch % lr_decay_step == 0 and epoch != 0  and loss_idx != 0.:
            if best_loss<(loss_mean/loss_idx):
                c_cnt += 1
                if c_cnt>5:
                    init_lr = init_lr*0.96
                    lr = init_lr
                    set_learning_rate(optimizer, init_lr)
                    c_cnt = 0

            else:
                best_loss = (loss_mean/loss_idx)
                c_cnt = 0

        loss_mean = 0.
        loss_idx = 0.

        for i, (imagesx, targetsx) in enumerate(dataloader):# load train data
            if is_train == False:
                print('len : ',len(imagesx), len(targetsx))

                for j in range(len(imagesx)):
                    im_ = imagesx[j].numpy()
                    img_show_ = (im_.transpose(1, 2, 0)+rgb_means).astype(np.uint8)

                    print('images : ',img_show_.shape)
                    print('targets : ',targetsx[j])

                    targetsx_ = targetsx[j].numpy()

                    image_name = str(img_idx)+'.jpg'
                    xml_name = image_name.replace('.jpg','.xml')
                    img_h, img_w = img_show_.shape[0],img_show_.shape[1]

                    for k in range(len(targetsx_)):
                        x1 = int(targetsx_[k][0]*img_size[1])
                        y1 = int(targetsx_[k][1]*img_size[0])
                        x2 = int(targetsx_[k][2]*img_size[1])
                        y2 = int(targetsx_[k][3]*img_size[0])
                        label_ = int(targetsx_[k][4])
                        print('x1,y1,x2,y2 --->>> : ',x1,y1,x2,y2)
                        bbox_ = (int(x1),int(y1),int(x2),int(y2))

                        plot_one_box(bbox_, img_show_, color=(255,0,0), label=labels_name[label_-1])


                    img_idx += 1

                    cv2.waitKey(1)
                    cv2.namedWindow('image',0)
                    cv2.imshow('image',img_show_)

                    cv2.waitKey(200)
            elif is_train == True:

                imagesx = torch.stack(imagesx, 0)
                if use_cuda:
                    images = Variable(imagesx.cuda().float())
                    targets = [Variable(anno.cuda().float()) for anno in targetsx]
                else:
                    images = Variable(imagesx.float())
                    targets = [Variable(anno.float()) for anno in targetsx]
                # forward
                load_t0 = time.time()
                out = model_(images)
                optimizer.zero_grad()
                loss_l, loss_c = criterion(out, priors, targets)
                loss = loss_l + loss_c
                loss_mean += (loss_l.item()+loss_c.item())
                loss_idx += 1.
                loss.backward()
                optimizer.step()
                t1 = time.time()

                load_t1 = time.time()
                if i % 10 == 0:
                    print('   %s - epoch (%3d/%s): '%('RFB_Net_vgg',i,int(dataset.__len__()/batch_size)),epoch,' total loss mean : %.6f - loss_l : %.6f - loss_c : %.6f'%(loss_mean/loss_idx,loss_l.item(),loss_c.item()),' lr : %.5f'%init_lr,' bs : ',batch_size,\
                    ' img_size : %s x %s'%(img_size[0],img_size[1]),' best_loss : %.6f'%(best_loss))

                    time.sleep(3)
                if (i % 50 == 0):
                    chkpt = {'epoch': epoch,'init_lr':init_lr,
                        'model': model_.state_dict()}
                    torch.save(chkpt, save_model_dir+'RFB_vgg_VOC_latest.pt')
                    # torch.save(net.state_dict(), args.save_folder+args.version+'_'+args.dataset + '_latest'+
                    #     '.pth')
        if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 ==0 and epoch > 200):
            torch.save(model_.state_dict(), save_model_dir+ 'epoches_'+repr(epoch) + '.pth')
    cv2.destroyAllWindows()
    print('\nwell done ~')
