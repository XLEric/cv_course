#-*-coding:utf-8-*-
# date:2019-11-18
# Author: xiang li
# function: RFBNet detect evaluation

import sys
sys.path.append('./')
import os
import time
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import numpy as np
from layers.functions import Detect,PriorBox
from models.RFB_Net_vgg import build_net
import cv2

from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from models.mobilenetv2 import MobileNetV2
from utils.common_utils import *
from acc_model import acc_model,acc_landmarks_model

#RFB CONFIGS
VOC_300 = {
    'num_classes': 21,
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

def nms(dets, thresh):
    """Pure Python NMS baseline."""
    # print('kkkkkkkkk  nms')
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

# VOC_CLASSESXXX = ('__background__','person', 'aeroplane', 'tvmonitor', 'train', 'boat', 'dog', 'chair', 'bird',\
#  'bicycle', 'bottle', 'sheep', 'diningtable', 'horse', 'motorbike', 'sofa', 'cow', 'car', 'cat', 'bus', 'pottedplant')
VOC_CLASSESXXX = ('__background__','face')
def detect_faces(ops,detect_faces,img_raw):
    sr_time = time.time()
    img_o = cv2.resize(img_raw, (ops.img_dim,ops.img_dim), interpolation = cv2.INTER_LINEAR)

    img_o = img_o-rgb_means
    img_o = img_o.transpose(2, 0, 1)
    er_time = time.time()
    img_o = torch.from_numpy(img_o)

    x = img_o.unsqueeze(0)
    if use_cuda:
        x = x.cuda()

    out = detect_model(x.float())# forward
    boxes, scores = detector.forward(out,priors)# decode

    boxes = boxes[0]
    scores = scores[0]

    boxes = boxes.cpu().detach().numpy()
    scores = scores.cpu().detach().numpy()
    boxes *= scale
    # boxes = boxes.cpu().detach().numpy()
    # scores = scores.cpu().detach().numpy()

    dets = []

    for j in range(1, num_classes):
        inds = np.where(scores[:, j] > ops.vis_thres)[0]

        c_bboxes = boxes[inds]
        c_scores = scores[inds, j]
        c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
            np.float32, copy=False)

        keep = nms(c_dets, ops.nms_threshold)
        c_dets = c_dets[keep, :]

        print('len(c_dets) : ',len(c_dets))
        for jj in range(len(c_dets)):
            x1_,y1_,x2_,y2_,score_ = c_dets[jj]
            bbox_ = (x1_,y1_,x2_,y2_)
            if score_>ops.vis_thres:
                dets.append([int(x1_),int(y1_),int(x2_),int(y2_),score_])
                # plot_one_box(bbox_, img_raw, color=(255,0,255), label=VOC_CLASSESXXX[j]+'_'+('%.3f')%score_)

    print('                              --> cost detail : {} '.format(er_time-sr_time))

    return dets

def get_faces_batch_landmarks(ops,dets,img_raw,use_cuda,draw_bbox = True):
    # 绘制图像
    image_batch = None
    r_bboxes = []
    imgs_crop = []
    for b in dets:
        if b[4] < ops.vis_thres:
            continue
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))

        r_bbox = refine_face_bbox((b[0],b[1],b[2],b[3]),img_raw.shape)
        r_bboxes.append(r_bbox)
        img_crop = img_raw[r_bbox[1]:r_bbox[3],r_bbox[0]:r_bbox[2]]
        imgs_crop.append(img_crop)
        img_ = cv2.resize(img_crop, (ops.landmarks_img_size[1],ops.landmarks_img_size[0]), interpolation = cv2.INTER_LINEAR) # INTER_LINEAR INTER_CUBIC

        img_ = img_.astype(np.float32)
        img_ = (img_-128.)/256.

        img_ = img_.transpose(2, 0, 1)
        img_ = np.expand_dims(img_,0)

        if image_batch is None:
            image_batch = img_
        else:
            image_batch = np.concatenate((image_batch,img_),axis=0)
    for b in dets:
        if b[4] < ops.vis_thres:
            continue
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        if draw_bbox:
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] - 3
        cv2.putText(img_raw, text, (cx, cy),cv2.FONT_HERSHEY_DUPLEX, 0.6, (155, 155, 255),3)
        cv2.putText(img_raw, text, (cx, cy),cv2.FONT_HERSHEY_DUPLEX, 0.6, (155, 10, 10),1)

    # 填充最大 关键点 批次数据
    if len(dets) < ops.max_batch_size:
        im_mask = np.zeros([1,3,ops.landmarks_img_size[0],ops.landmarks_img_size[1]], dtype = np.float32)
        for i in range(ops.max_batch_size-len(dets)):
            if image_batch is None:
                image_batch = im_mask
            else:
                image_batch = np.concatenate((image_batch,im_mask),axis=0)

    image_batch = torch.from_numpy(image_batch).float()

    if use_cuda:
        image_batch = image_batch.cuda()  # (bs, 3, h, w)

    pre_ = landmarks_model(image_batch.float())
    # print(pre_.size())
    output = pre_.cpu().detach().numpy()
    # print('output shape : ',output.shape)
    # n_array = np.zeros([ops.landmarks_img_size[0],ops.landmarks_img_size[1],3], dtype = np.float)
    for i in range(len(dets)):

        dict_landmarks = draw_landmarks(imgs_crop[i],output[i],draw_circle = False)

        draw_contour(img_raw,dict_landmarks,r_bboxes[i])

    cv2.putText(img_raw, 'face:'+str(len(dets)), (3,35),cv2.FONT_HERSHEY_DUPLEX, 1.45, (55, 255, 255),5)
    cv2.putText(img_raw, 'face:'+str(len(dets)), (3,35),cv2.FONT_HERSHEY_DUPLEX, 1.45, (135, 135, 5),2)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=' FRBNet Inferece')
    # FaceBoxes_epoch_290
    parser.add_argument('-m', '--detect_model', default='./model_detect_dir/epoches_60.pth',
                        type=str, help='detect_model')
    parser.add_argument('--GPUS', type=str, default = '0',help = 'GPUS') # GPU选择
    parser.add_argument('--img_dim', type=int,default=300, help='img_dim')
    parser.add_argument('--num_classes', type=int,default=2, help='num_classes')

    parser.add_argument('--nms_threshold', type=float,default=0.3,  help='nms_threshold')
    parser.add_argument('--vis_thres',type=float, default=0.45,  help='visualization_threshold')

    #-----------------------------------------------------------------------------------------
    parser.add_argument('--landmarks_model', type=str, default = './landmarks_model/resnet50_epoch-2303.pth',
        help = 'landmarks_model') # 模型路径
    parser.add_argument('--landmarks_network', type=str, default = 'resnet_50',
        help = 'model : resnet_18,resnet_34,resnet_50,resnet_101,resnet_152,mobilenetv2') # 模型类型
    parser.add_argument('--landmarks_num_classes', type=int , default = 196,
        help = 'landmarks_num_classes') #  分类类别个数
    parser.add_argument('--landmarks_img_size', type=tuple , default = (256,256),
        help = 'landmarks_img_size') # 输入landmarks 模型图片尺寸
    #-----------------------------------------------------------------------------------------
    parser.add_argument('--force_cpu', type=bool, default = False,
        help = 'force_cpu') # 前向推断硬件选择
    parser.add_argument('--max_batch_size', type=int , default = 1,
        help = 'max_batch_size') #  最大 landmarks - max_batch_size

    parser.add_argument('--test_path', type=str, default = '../chapter_07/video/rw_7.mp4',
        help = 'test_path') # 测试文件路径
    #--------------------------------------------------------------------------

    print('\n/******************* {} ******************/\n'.format(parser.description))
    #--------------------------------------------------------------------------
    ops = parser.parse_args()# 解析添加参数
    #--------------------------------------------------------------------------
    print('----------------------------------')

    unparsed = vars(ops) # parse_args()方法的返回值为namespace，用vars()内建函数化为字典
    for key in unparsed.keys():
        print('{} : {}'.format(key,unparsed[key]))
    use_cuda = torch.cuda.is_available()
    os.environ['CUDA_VISIBLE_DEVICES'] = ops.GPUS
    # torch.set_num_threads(1)
    if use_cuda:
        cudnn.benchmark = True
    #---------------------------------------------------------------- 构建 landmarks 模型
    if ops.landmarks_network == 'resnet_18':
        landmarks_model=resnet18(num_classes=ops.landmarks_num_classes, img_size=ops.landmarks_img_size[0])
    elif ops.landmarks_network == 'resnet_34':
        landmarks_model=resnet34(num_classes=ops.landmarks_num_classes, img_size=ops.landmarks_img_size[0])
    elif ops.landmarks_network == 'resnet_50':
        landmarks_model=resnet50(num_classes=ops.landmarks_num_classes, img_size=ops.landmarks_img_size[0])
    elif ops.landmarks_network == 'resnet_101':
        landmarks_model=resnet101(num_classes=ops.landmarks_num_classes, img_size=ops.landmarks_img_size[0])
    elif ops.landmarks_network == 'resnet_152':
        landmarks_model=resnet152(num_classes=ops.landmarks_num_classes, img_size=ops.landmarks_img_size[0])
    elif ops.landmarks_network == 'mobilenetv2':
        landmarks_model=MobileNetV2(n_class =ops.landmarks_num_classes, input_size=ops.landmarks_img_size[0])
    else:
        print('error no the struct model : {}'.format(ops.model))

    device = torch.device("cuda:0" if use_cuda else "cpu")
    # 加载测试模型
    if os.access(ops.landmarks_model,os.F_OK):# checkpoint
        # chkpt = torch.load(ops.landmarks_model, map_location=device)
        # landmarks_model.load_state_dict(chkpt)

        chkpt = torch.load(ops.landmarks_model, map_location=lambda storage, loc: storage)
        landmarks_model.load_state_dict(chkpt)
        landmarks_model.eval() # 设置为前向推断模式
        print('load landmarks model : {}'.format(ops.landmarks_model))
        print('\n/******************* landmarks model acc  ******************/')
        acc_landmarks_model(ops,landmarks_model)
    landmarks_model = landmarks_model.to(device)

    #--------------------------------------------------------------------------- 构建人脸检测模型
    cfg = VOC_300
    rgb_means = VOC_300['rgb_means']
    num_classes = ops.num_classes
    use_cuda = torch.cuda.is_available()
    detect_model = build_net('test', ops.img_dim, ops.num_classes)    # initialize detector

    #---------------------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # chkpt = torch.load(ops.detect_model, map_location=device)
    chkpt = torch.load(ops.detect_model, map_location=lambda storage, loc: storage)
    detect_model.load_state_dict(chkpt)
    detect_model.eval() # 设置为前向推断模式

    acc_model(ops,detect_model)
    detect_model = detect_model.to(device)

    detector = Detect(ops.num_classes,0,cfg)#  num_classes, bkg_label, cfg
    priorbox = PriorBox(cfg,debug_ = False)
    with torch.no_grad():
        priors = priorbox.forward()
        if use_cuda:
            priors = priors.cuda()

    video_capture = cv2.VideoCapture(ops.test_path)
    ret, img_raw = video_capture.read()
    if ret:
        # scale = torch.Tensor([img_raw.shape[1], img_raw.shape[0],img_raw.shape[1], img_raw.shape[0]])
        scale = [img_raw.shape[1], img_raw.shape[0],img_raw.shape[1], img_raw.shape[0]]
        # if use_cuda:
        #     scale = scale.cuda()
    else:
        print('--------------- error read video_capture ')

    with torch.no_grad():
        idx  = 0
        while True:
            ret, img_raw = video_capture.read()

            if ret:
                if idx == 0:
                    print('video shape : {}'.format(img_raw.shape))
                idx += 1
                if idx%2!=0:
                    continue

                s_time = time.time()
                dets = detect_faces(ops,detect_faces,img_raw)
                er_time = time.time()
                time.sleep(0.001)
                get_faces_batch_landmarks(ops,dets,img_raw,use_cuda,draw_bbox = True)
                er2_time = time.time()
                if (er_time-s_time)>0.1:
                    print('{} * detect_faces cost : {} '.format(idx,er_time-s_time))
                # print(' * landmarks cost : {} '.format(er2_time-s_time))

                e_time = time.time()
                str_fps = ('{:.2f} Fps'.format(1./(e_time-s_time)))
                cv2.putText(img_raw, str_fps, (5,img_raw.shape[0]-3),cv2.FONT_HERSHEY_DUPLEX, 1.6, (255, 0, 255),4)
                cv2.putText(img_raw, str_fps, (5,img_raw.shape[0]-3),cv2.FONT_HERSHEY_DUPLEX, 1.6, (255, 255, 0),1)
                print(str_fps)
                cv2.namedWindow('video',0)
                cv2.imshow('video',img_raw)

                if cv2.waitKey(1) == 27 :
                    break
    cv2.destroyAllWindows()
