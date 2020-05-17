#-*-coding:utf-8-*-
# date:2020-05-02
# Author: xiang li

from __future__ import print_function
import os
import sys
import time
sys.path.append('./')
import cv2
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms

from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm

from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from models.mobilenetv2 import MobileNetV2
from utils.common_utils import *

from thop import profile

from thop import clever_format # 增加可读性

def detect_faces(ops,detect_model,img_raw):
    img = np.float32(img_raw)
    if ops.resize != 1:
        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= ops.color_mean
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    loc, conf, landms = detect_model(img)  # forward pass

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / ops.resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / ops.resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > ops.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:ops.keep_top_k]
    # order = scores.argsort()[::-1]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, ops.nms_threshold)

    dets = dets[keep, :]
    landms = landms[keep]

    dets = np.concatenate((dets, landms), axis=1)

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

def detetc_landmarks(b,landmarks_model,img_raw):
    r_bbox = refine_face_bbox((b[0],b[1],b[2],b[3]),img_o.shape)
    img_crop = img_o[r_bbox[1]:r_bbox[3],r_bbox[0]:r_bbox[2]]
    img_ = cv2.resize(img_crop, (ops.landmarks_img_size[1],ops.landmarks_img_size[0]), interpolation = cv2.INTER_CUBIC)

    img_ = img_.astype(np.float32)
    img_ = (img_-128.)/256.

    img_ = img_.transpose(2, 0, 1)
    img_ = torch.from_numpy(img_)
    img_ = img_.unsqueeze_(0)

    if use_cuda:
        img_ = img_.cuda()  # (bs, 3, h, w)

    pre_ = landmarks_model(img_.float())
    # print(pre_.size())
    output = pre_.cpu().detach().numpy()
    output = np.squeeze(output)
    dict_landmarks = draw_landmarks(img_crop,output,draw_circle = False)

    draw_contour(img_raw,dict_landmarks,r_bbox)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=' Retinaface Inferece')
    # './weights_mobile/mobilenet0.25_Final.pth'
    # './weights_resnet/Resnet50_epoch_105.pth'
    parser.add_argument('--detect_model', type=str, default = './weights_resnet/Resnet50_epoch_200.pth',
        help = 'detect_model_path') # 模型类型
    parser.add_argument('--detect_network',  type=str, default='resnet50',
        help='Backbone network mobile0.25 or resnet50')

    parser.add_argument('--color_mean', type=tuple , default = (104, 117, 123),
        help = 'color mean') # 模型输入图片颜色偏置
    parser.add_argument('--nms_threshold',  type=float, default=0.25,
        help='nms_threshold')
    parser.add_argument('--keep_top_k', type=int, default=25,
        help='keep_top_k')
    parser.add_argument('--confidence_threshold', type=float,default=0.85,
        help='confidence_threshold')
    parser.add_argument('--vis_thres', type=float, default=0.85,
        help='visualization_threshold')

    parser.add_argument('--resize', type=float, default=1., help='resize')

    parser.add_argument('--GPUS', type=str, default = '0',
        help = 'GPUS') # GPU选择
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
    parser.add_argument('--max_batch_size', type=int , default = 3,
        help = 'max_batch_size') #  最大 landmarks - max_batch_size
    parser.add_argument('--test_path', type=str, default = './video/jk_1.mp4',
        help = 'test_path') # 测试文件路径

    print('\n/******************* {} ******************/\n'.format(parser.description))
    #--------------------------------------------------------------------------
    ops = parser.parse_args()# 解析添加参数
    #--------------------------------------------------------------------------
    print('----------------------------------')

    unparsed = vars(ops) # parse_args()方法的返回值为namespace，用vars()内建函数化为字典
    for key in unparsed.keys():
        print('{} : {}'.format(key,unparsed[key]))

    os.environ['CUDA_VISIBLE_DEVICES'] = ops.GPUS
    use_cuda = torch.cuda.is_available()
    torch.set_num_threads(1)
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

    # dummy_input = torch.randn(1, 3, ops.landmarks_img_size[0], ops.landmarks_img_size[1])
    # flops, params = profile(landmarks_model, inputs=(dummy_input, ))
    # flops, params = clever_format([flops, params], "%.3f")

    # print('flops ： {} , params : {}'.format(flops, params))


    device = torch.device("cuda:0" if use_cuda else "cpu")
    landmarks_model = landmarks_model.to(device)
    landmarks_model.eval() # 设置为前向推断模式



    # 加载测试模型
    if os.access(ops.landmarks_model,os.F_OK):# checkpoint
        chkpt = torch.load(ops.landmarks_model, map_location=device)
        landmarks_model.load_state_dict(chkpt)
        print('load landmarks model : {}'.format(ops.landmarks_model))

    #--------------------------------------------------------------------------- 构建人脸检测模型
    cfg = None
    if ops.detect_network == "mobile0.25":
        cfg = cfg_mnet
    elif ops.detect_network == "resnet50":
        cfg = cfg_re50
    # net and model
    detect_model = RetinaFace(cfg=cfg, phase = 'test')

    detect_model = detect_model.to(device)

    if os.access(ops.detect_model,os.F_OK):# checkpoint
        chkpt = torch.load(ops.detect_model, map_location=device)
        detect_model.load_state_dict(chkpt)
        print('load detect model : {}'.format(ops.detect_model))

    detect_model.eval()
    if use_cuda:
        cudnn.benchmark = True

    print('loading model done ~')
    #-------------------------------------------------------------------------- run vedio
    video_capture = cv2.VideoCapture(ops.test_path)
    with torch.no_grad():
        idx  = 0
        while True:
            ret, img_raw = video_capture.read()

            if ret:
                if idx == 0:
                    print('video shape : {}'.format(img_raw.shape))
                idx += 1

                if idx%3!=0:
                    continue
                s_time = time.time()

                dets = detect_faces(ops,detect_model,img_raw)

                get_faces_batch_landmarks(ops,dets,img_raw,use_cuda)

                e_time = time.time()
                str_fps = ('{:.2f} Fps'.format(1./(e_time-s_time)))
                cv2.putText(img_raw, str_fps, (5,img_raw.shape[0]-3),cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 0, 255),4)
                cv2.putText(img_raw, str_fps, (5,img_raw.shape[0]-3),cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 0),1)
                    #---------------------------------------------------------------------------
                cv2.namedWindow('video',0)
                cv2.imshow('video',img_raw)
                if cv2.waitKey(1) == 27:
                    break
            else:
                break

        cv2.destroyAllWindows()
