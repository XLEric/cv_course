#-*-coding:utf-8-*-
# date:2020-05-10
# Author: xiang li

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import numpy as np
import sys
import time
sys.path.append('./')

from opts import opts
from detectors.detector_factory import detector_factory

from landmarks_models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from landmarks_models.mobilenetv2 import MobileNetV2
from landmarks_utils.common_utils import *
import torch
import torch.backends.cudnn as cudnn
from lib.models.acc_model import acc_model

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
def get_faces_batch_landmarks(ops,landmarks_model,dets,img_raw,use_cuda,draw_bbox = True):
    vis_thres = 0.5
    # 绘制图像
    image_batch = None
    r_bboxes = []
    imgs_crop = []
    for b in dets:
        if b[4] < vis_thres:
            continue
        text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        if ((b[3]-b[1])*(b[2]-b[0]))<100:
            continue
        r_bbox = refine_face_bbox((b[0],b[1],b[2],b[3]),img_raw.shape)
        r_bboxes.append(r_bbox)
        img_crop = img_raw[r_bbox[1]:r_bbox[3],r_bbox[0]:r_bbox[2]]
        imgs_crop.append(img_crop)
        img_ = cv2.resize(img_crop, (256,256), interpolation = cv2.INTER_LINEAR) # INTER_LINEAR INTER_CUBIC

        img_ = img_.astype(np.float32)
        img_ = (img_-128.)/256.

        img_ = img_.transpose(2, 0, 1)
        img_ = np.expand_dims(img_,0)

        if image_batch is None:
            image_batch = img_
        else:
            image_batch = np.concatenate((image_batch,img_),axis=0)
    for b in dets:
        if b[4] < 0.5:
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
    if len(r_bboxes) < 5:
        im_mask = np.zeros([1,3,256,256], dtype = np.float32)
        for i in range(5-len(r_bboxes)):
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
    # print(output)
    # n_array = np.zeros([ops.landmarks_img_size[0],ops.landmarks_img_size[1],3], dtype = np.float)
    for i in range(len(r_bboxes)):

        dict_landmarks = draw_landmarks(imgs_crop[i],output[i],draw_circle = False)
        # cv2.imshow('imgs_crop',imgs_crop[i])

        draw_contour(img_raw,dict_landmarks,r_bboxes[i])

    cv2.putText(img_raw, 'face:'+str(len(dets)), (3,35),cv2.FONT_HERSHEY_DUPLEX, 1.45, (55, 255, 255),5)
    cv2.putText(img_raw, 'face:'+str(len(dets)), (3,35),cv2.FONT_HERSHEY_DUPLEX, 1.45, (135, 135, 5),2)
def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  #---------------------------------------------------------------- 构建 landmarks 模型
  landmarks_model_pth  = './landmarks_model/resnet-18_epoch-2300.pth'
  landmarks_img_size = 256
  landmarks_network = 'resnet_18'
  landmarks_num_classes = 196
  if landmarks_network == 'resnet_18':
      landmarks_model=resnet18(num_classes=landmarks_num_classes, img_size=landmarks_img_size)
  elif landmarks_network == 'resnet_34':
      landmarks_model=resnet34(num_classes=landmarks_num_classes, img_size=landmarks_img_size)
  elif landmarks_network == 'resnet_50':
      landmarks_model=resnet50(num_classes=landmarks_num_classes, img_size=landmarks_img_size)
  elif landmarks_network == 'resnet_101':
      landmarks_model=resnet101(num_classes=landmarks_num_classes, img_size=landmarks_img_size)
  elif landmarks_network == 'resnet_152':
      landmarks_model=resnet152(num_classes=landmarks_num_classes, img_size=landmarks_img_size)
  elif landmarks_network == 'mobilenetv2':
      landmarks_model=MobileNetV2(n_class =landmarks_num_classes, input_size=landmarks_img_size)
  else:
      print('error no the struct model : {}'.format(landmarks_model))
  use_cuda = torch.cuda.is_available()

  torch.set_num_threads(1)
  if use_cuda:
      cudnn.benchmark = True

  device = torch.device("cuda:0" if use_cuda else "cpu")


  # 加载测试模型
  if os.access(landmarks_model_pth,os.F_OK):# checkpoint
      # chkpt = torch.load(ops.landmarks_model, map_location=device)
      # landmarks_model.load_state_dict(chkpt)

      chkpt = torch.load(landmarks_model_pth, map_location=lambda storage, loc: storage)
      landmarks_model.load_state_dict(chkpt)
      landmarks_model.eval() # 设置为前向推断模式
      print('load landmarks model : {}'.format(landmarks_model_pth))
      print('\n/******************* landmarks model acc  ******************/')
      acc_model(None,landmarks_model)
  landmarks_model = landmarks_model.to(device)


  #--------------------------------------------------------------

  opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)

  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
    detector.pause = False
    font = cv2.FONT_HERSHEY_SIMPLEX
    idx_ = 0
    while True:
        _, img = cam.read()
        idx_ += 1
        if idx_%2==0:
            continue
        # cv2.namedWindow('input',0)
        # cv2.imshow('input', img)
        s_time = time.time()
        img_,ret,faces_boxes,person_boxes = detector.run(img)# base_detector
        get_faces_batch_landmarks(None,landmarks_model,faces_boxes,img,use_cuda,draw_bbox = True)
        for i in range(len(person_boxes)):
            x1,y1,x2,y2,score = person_boxes[i]
            cv2.rectangle(img,(x1,y1),(x2,y2),(30,255,255),2)
        # print(img.shape)
        # cv2.putText(img,'MultiPose fps:{:.2f}'.format(1./ret['tot']),(10,img.shape[0]-10),font,2.0,(185,55,255),12)
        # cv2.putText(img,'MultiPose fps:{:.2f}'.format(1./ret['tot']),(10,img.shape[0]-10),font,2.0,(185,255,55),3)
        e_time = time.time()
        str_fps = ('{:.2f} Fps'.format(1./(e_time-s_time)))
        cv2.putText(img, str_fps, (5,img.shape[0]-3),cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 0, 255),4)
        cv2.putText(img, str_fps, (5,img.shape[0]-3),cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 0),1)

        img_s = np.hstack((img,img_))
        cv2.line(img_s,(int(img_s.shape[1]/2),0),(int(img_s.shape[1]/2),int(img_s.shape[0])),(25,90,255),20)
        cv2.line(img_s,(int(img_s.shape[1]/2),0),(int(img_s.shape[1]/2),int(img_s.shape[0])),(120,120,120),8)

        cv2.namedWindow('img_s',0)
        cv2.imshow('img_s', img_s)

        time_str = ''
        for stat in time_stats:
          time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        # print(time_str)
        if cv2.waitKey(1) == 27:
            return  # esc to quit
  else:
    print('------------->>.')
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      image_names = [opt.demo]

    for (image_name) in image_names:

      ret = detector.run(image_name)
      time_str = ''
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      # print(time_str)
if __name__ == '__main__':
  opt = opts().init()
  if 1:
      demo(opt)
  # except:
  #     pass
