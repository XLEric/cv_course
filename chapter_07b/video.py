#-*-coding:utf-8-*-
# date:2020-05-02
# Author: xiang li

from __future__ import print_function
import os
import sys
import time
sys.path.append('./')
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from config import cfg
from layers.functions.prior_box import PriorBox

import cv2
from models.faceboxes import FaceBoxes
from utils.box_utils import decode
from utils.timer import Timer

from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from models.mobilenetv2 import MobileNetV2
from utils.common_utils import *

parser = argparse.ArgumentParser(description='FaceBoxes')

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
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


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def detect_faces(ops,detect_faces,img_raw):
    img = np.float32(img_raw)
    if resize != 1:
        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)


    loc, conf = detect_model(img)  # forward pass

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

    # ignore low scores
    inds = np.where(scores > ops.confidence_threshold)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:ops.top_k]
    boxes = boxes[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    #keep = py_cpu_nms(dets, ops.nms_threshold)
    # keep = nms(dets, ops.nms_threshold,force_cpu=True)
    keep = py_cpu_nms(dets, ops.nms_threshold)
    dets = dets[keep, :]

    # keep top-K faster NMS
    dets = dets[:ops.keep_top_k, :]

    return dets
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
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=' FaceBoxes Inferece')
    parser.add_argument('-m', '--detect_model', default='weights/FaceBoxes_epoch_160.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--GPUS', type=str, default = '0',help = 'GPUS') # GPU选择
    parser.add_argument('--save_folder', default='eval/', type=str, help='Dir to save results')
    parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
    parser.add_argument('--confidence_threshold', default=0.78, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=200, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.25, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=200, type=int, help='keep_top_k')
    parser.add_argument('--vis_thres', default=0.78, type=float, help='visualization_threshold')
    #-----------------------------------------------------------------------------------------
    parser.add_argument('--landmarks_model', type=str, default = './landmarks_model/model_epoch-1800.pth',
        help = 'landmarks_model') # 模型路径
    parser.add_argument('--landmarks_network', type=str, default = 'resnet_50',
        help = 'model : resnet_18,resnet_34,resnet_50,resnet_101,resnet_152,mobilenetv2') # 模型类型
    parser.add_argument('--landmarks_num_classes', type=int , default = 196,
        help = 'landmarks_num_classes') #  分类类别个数
    parser.add_argument('--landmarks_img_size', type=tuple , default = (256,256),
        help = 'landmarks_img_size') # 输入landmarks 模型图片尺寸
    #-----------------------------------------------------------------------------------------
    parser.add_argument('--test_path', type=str, default = '../chapter_07/video/jk_1.mp4',
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

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda:0" if use_cuda else "cpu")
    landmarks_model = landmarks_model.to(device)
    landmarks_model.eval() # 设置为前向推断模式

    # 加载测试模型
    if os.access(ops.landmarks_model,os.F_OK):# checkpoint
        chkpt = torch.load(ops.landmarks_model, map_location=device)
        landmarks_model.load_state_dict(chkpt)
        print('load landmarks model : {}'.format(ops.landmarks_model))

    #--------------------------------------------------------------------------- 构建人脸检测模型
    # detect_model
    detect_model = FaceBoxes(phase='test', size=None, num_classes=2)    # initialize detector
    detect_model = load_model(detect_model, ops.detect_model, ops.cpu)
    detect_model.eval()
    print('Finished loading model!')
    print(detect_model)
    cudnn.benchmark = True

    detect_model = detect_model.to(device)

    video_capture = cv2.VideoCapture(ops.test_path)

    resize = 1
    with torch.no_grad():
        idx  = 0
        while True:
            ret, img_o = video_capture.read()


            if ret:
                if idx == 0:
                    print('video shape : {}'.format(img_o.shape))
                    scale = 800/float(img_o.shape[1])
                idx += 1
                if idx%7!=0:
                    continue
                s_time = time.time()

                # img_o = cv2.resize(img_o, (int(img_o.shape[1]*scale),int(img_o.shape[0]*scale)), interpolation=cv2.INTER_LINEAR)

                img_raw = img_o.copy()

                dets = detect_faces(ops,detect_model,img_raw)

                # 绘制图像
                for b in dets:
                    if b[4] < ops.vis_thres:
                        continue
                    text = "{:.4f}".format(b[4])
                    b = list(map(int, b))
                    cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                    cx = b[0]
                    cy = b[1] - 3
                    cv2.putText(img_raw, text, (cx, cy),
                                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                    detetc_landmarks(b,landmarks_model,img_raw)

                e_time = time.time()
                str_fps = ('{:.2f} Fps'.format(1./(e_time-s_time)))
                # cv2.putText(img_raw, str_fps, (5,img_raw.shape[0]-3),cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 0, 255),4)
                # cv2.putText(img_raw, str_fps, (5,img_raw.shape[0]-3),cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 0),1)

                cv2.namedWindow('video',0)
                cv2.imshow('video',img_raw)
                if cv2.waitKey(1) == 27:
                    break
            else:
                break
    cv2.destroyAllWindows()
