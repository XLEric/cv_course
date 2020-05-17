#-*-coding:utf-8-*-
# date:2020-05-13
# Author: xiang li
# function: LFFD detect evaluation

import argparse
import time
import os
import sys
sys.path.append('..')
sys.path.append('../..')
import torch
import torch.backends.cudnn as cudnn
from net_farm.naivenet import naivenet20
from acc_model import acc_model
import math
import numpy as np
import cv2

from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from models.mobilenetv2 import MobileNetV2
from utils.common_utils import *

# 非形变处理
def letterbox(img_,size_=416,mean_rgb = (0,0,0)):

    shape_ = img_.shape[:2]  # shape = [height, width]
    ratio = float(size_) / max(shape_)  # ratio  = old / new
    new_shape_ = (round(shape_[1] * ratio), round(shape_[0] * ratio))
    dw_ = (size_ - new_shape_[0]) / 2  # width padding
    dh_ = (size_ - new_shape_[1]) / 2  # height padding
    top_, bottom_ = round(dh_ - 0.1), round(dh_ + 0.1)
    left_, right_ = round(dw_ - 0.1), round(dw_ + 0.1)
    # resize img
    img_a = cv2.resize(img_, new_shape_, interpolation=cv2.INTER_LINEAR)

    img_a = cv2.copyMakeBorder(img_a, top_, bottom_, left_, right_, cv2.BORDER_CONSTANT, value=mean_rgb)  # padded square
    # print('fix size : ',img_a.shape)
    return img_a

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

def scale_coords(img_size, coords, img0_shape):# image size 转为 原图尺寸
    # Rescale x1, y1, x2, y2 from 416 to image size
    # print('coords     : ',coords)
    # print('img0_shape : ',img0_shape)
    gain = float(img_size) / max(img0_shape)  # gain  = old / new
    # print('gain       : ',gain)
    pad_x = (img_size - img0_shape[1] * gain) / 2  # width padding
    pad_y = (img_size - img0_shape[0] * gain) / 2  # height padding
    # print('pad_xpad_y : ',pad_x,pad_y)
    coords[:, [0, 2]] -= pad_x
    coords[:, [1, 3]] -= pad_y
    coords[:, :4] /= gain
    coords[:, :4] = np.clip(coords[:, :4], 0,max(img0_shape[0],img0_shape[1]))# 夹紧区间最小值不为负数
    return coords
def detect_faces(ops,detect_model,img_raw):
    img = letterbox(img_raw,size_=ops.input_height,mean_rgb = (0,0,0))
    # img = cv2.resize(img_raw, (ops.input_width,ops.input_height), interpolation=cv2.INTER_LINEAR)

    im_input = img.astype(np.float32)
    im_input = (im_input - 127.5) / 127.5
    im_input = im_input.transpose(2, 0, 1)
    im_input = torch.from_numpy(im_input)
    im_input = im_input.unsqueeze_(0)

    if use_cuda:
        im_input = im_input.cuda()  # (bs, 3, h, w)

    results = detect_model(im_input.float())
    dets = []
    outputs = []
    for i in range(len(results)):
        # print('{}) {}'.format(i+1,results[i].size()))
        if i%2 ==0:
            outputs.append(torch.softmax(results[i],dim=1).cpu().detach().numpy())
        else:
            outputs.append(results[i].cpu().detach().numpy())

    for i in range(ops.num_output_scales):

        # score_map = np.squeeze(outputs[i * 2]).transpose(1,2,0)[:,:,0]*(1.-np.squeeze(outputs[i * 2]).transpose(1,2,0)[:,:,1])
        score_map = np.squeeze(outputs[i * 2]).transpose(1,2,0)[:,:,0]
        # score_map = 1.-np.squeeze(outputs[i * 2]).transpose(1,2,0)[:,:,1]
    #     print('score_map shape : ',score_map.shape)
        # score_map_show = score_map * 255
        # score_map_show[score_map_show < 0] = 0
        # score_map_show[score_map_show > 255] = 255
        # cv2.imshow('score_map' + str(i), cv2.resize(score_map_show.astype(dtype=np.uint8), (0, 0), fx=2, fy=2))
        # cv2.waitKey()

        bbox_map = np.squeeze(outputs[i * 2 + 1], 0)

        # print('bbox_map shape : ',bbox_map.shape)

        RF_center_Xs = np.array([ops.receptive_field_center_start[i] + ops.receptive_field_stride[i] * x for x in range(score_map.shape[1])])
        RF_center_Xs_mat = np.tile(RF_center_Xs, [score_map.shape[0], 1])
        RF_center_Ys = np.array([ops.receptive_field_center_start[i] + ops.receptive_field_stride[i] * y for y in range(score_map.shape[0])])
        RF_center_Ys_mat = np.tile(RF_center_Ys, [score_map.shape[1], 1]).T


        # receptive_field_centers = numpy.array(
        #     [self.receptive_field_center_start[i] + w * self.receptive_field_stride[i] for w in range(self.feature_map_size_list[i])])

        x_lt_mat = RF_center_Xs_mat - bbox_map[0, :, :] * constant[i]
        y_lt_mat = RF_center_Ys_mat - bbox_map[1, :, :] * constant[i]
        x_rb_mat = RF_center_Xs_mat - bbox_map[2, :, :] * constant[i]
        y_rb_mat = RF_center_Ys_mat - bbox_map[3, :, :] * constant[i]

        x_lt_mat = x_lt_mat / ops.resize_scale
        x_lt_mat[x_lt_mat < 0] = 0
        y_lt_mat = y_lt_mat / ops.resize_scale
        y_lt_mat[y_lt_mat < 0] = 0
        x_rb_mat = x_rb_mat / ops.resize_scale
        x_rb_mat[x_rb_mat > img.shape[1]] = img.shape[1]
        y_rb_mat = y_rb_mat / ops.resize_scale
        y_rb_mat[y_rb_mat > img.shape[0]] = img.shape[0]

        select_index = np.where(score_map > ops.vis_thres)

        for idx in range(select_index[0].size):
            # score_map
            dets.append([x_lt_mat[select_index[0][idx], select_index[1][idx]],
                                    y_lt_mat[select_index[0][idx], select_index[1][idx]],
                                    x_rb_mat[select_index[0][idx], select_index[1][idx]],
                                    y_rb_mat[select_index[0][idx], select_index[1][idx]],
                                    score_map[select_index[0][idx], select_index[1][idx]]])
    if len(dets) > 0:
        # print(dets)
        dets = np.array(dets)
        keep = py_cpu_nms(dets, ops.nms_thres)
        dets = dets[keep, :]

        dets[:, :4] = scale_coords(640, dets[:, :4], img_raw.shape).round()

        return list(dets)
    else:
        return []

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
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inferece')
    # ./configuration_10_320_20L_5scales_v2_2020-05-14-13-42-49/train_10_320_20L_5scales_v2/face_v2_100000.pth
    # ./configuration_10_320_20L_5scales_v2_2020-05-12-16-51-53/train_10_320_20L_5scales_v2/face_v2_248000.pth
    parser.add_argument('-m', '--detect_model', default='./configuration_10_320_20L_5scales_v2_2020-05-14-13-42-49/train_10_320_20L_5scales_v2/face_v2_100000.pth',
                        type=str, help='Trained state_dict file path to open')

    parser.add_argument('--GPUS', type=str, default = '0',help = 'GPUS') # GPU选择
    parser.add_argument('--vis_thres', default=0.65, type=float, help='visualization_threshold')
    parser.add_argument('--nms_thres', type=float, default = 0.4,help = 'nms_thres')
    parser.add_argument('--input_height', type=int, default = 640,help = 'input height for network')
    parser.add_argument('--input_width', type=int, default = 640,help = 'input width for network')
    parser.add_argument('--num_image_channel', type=int, default = 3,help = 'the number of image channels')
    parser.add_argument('--num_output_scales', type=int, default = 5,help = 'the number of output scales (loss branches)')
    parser.add_argument('--feature_map_size_list', type=list, default = [159, 79, 39, 19, 9],help = 'feature map size for each scale')
    parser.add_argument('--bbox_small_list', type=list, default = [10, 20, 40, 80, 160],help = 'bbox lower bound for each scale')
    parser.add_argument('--bbox_large_list', type=list, default = [20, 40, 80, 160, 320],help = 'bbox upper bound for each scale')
    parser.add_argument('--receptive_field_stride', type=list, default = [4, 8, 16, 32, 64],help = 'RF stride for each scale')
    parser.add_argument('--receptive_field_center_start', type=list, default =[3, 7, 15, 31, 63],help = 'the start location of the first RF of each scale')
    parser.add_argument('--num_output_channels', type=int, default =6,help = '2 channels for classification and 4 for bbox regression')
    parser.add_argument('--resize_scale', type=float, default =1.,help = 'resize_scale')

    #-----------------------------------------------------------------------------------------
    parser.add_argument('--landmarks_model', type=str, default = './landmarks_model/resnet-18_epoch-2300.pth',
        help = 'landmarks_model') # 模型路径
    parser.add_argument('--landmarks_network', type=str, default = 'resnet_18',
        help = 'model : resnet_18,resnet_34,resnet_50,resnet_101,resnet_152,mobilenetv2') # 模型类型
    parser.add_argument('--landmarks_num_classes', type=int , default = 196,
        help = 'landmarks_num_classes') #  分类类别个数
    parser.add_argument('--landmarks_img_size', type=tuple , default = (256,256),
        help = 'landmarks_img_size') # 输入landmarks 模型图片尺寸

    #---------------------------------------------------------------------------------------------------
    parser.add_argument('--max_batch_size', type=int , default = 2,
        help = 'max_batch_size') #  最大 landmarks - max_batch_size
    parser.add_argument('--test_path', type=str, default = 'H:/project/git_project/cv_course/chapter_07/video/rw_7.mp4',
        help = 'test_path') # 测试文件路径

    print('\n/******************* {} ******************/\n'.format(parser.description))
    #--------------------------------------------------------------------------
    ops = parser.parse_args()# 解析添加参数


    # bbox gray lower bound for each scale
    bbox_small_gray_list = [math.floor(v * 0.9) for v in ops.bbox_small_list]

    # here we use bbox_large_list for better regression
    receptive_field_list = ops.bbox_large_list

    # bbox gray upper bound for each scale
    bbox_large_gray_list = [math.ceil(v * 1.1) for v in ops.bbox_large_list]
    # the RF size of each scale used for normalization,


    constant = [i / 2.0 for i in receptive_field_list]

    assert len(ops.bbox_small_list) == ops.num_output_scales
    assert len(ops.bbox_large_list) == ops.num_output_scales
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

    detect_model = naivenet20()
    # print(detect_model)

    device = torch.device("cuda:0" if use_cuda else "cpu")

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
        acc_model(ops,landmarks_model)
    landmarks_model = landmarks_model.to(device)
    #--------------------------------------------------------------------------- 构建人脸检测模型
    # resize_scale = 1



    # 加载测试模型
    if os.access(ops.detect_model,os.F_OK):# checkpoint
        chkpt = torch.load(ops.detect_model, map_location=lambda storage, loc: storage)
        detect_model.load_state_dict(chkpt)
        detect_model.eval() # 设置为前向推断模式
        print('load detect  model : {}'.format(ops.detect_model))
        print('\n/******************* detect model acc  ******************/')
        acc_model(ops,detect_model)
    detect_model = detect_model.to(device)

    video_capture = cv2.VideoCapture(ops.test_path)

    # resize = 1
    with torch.no_grad():
        idx  = 0
        while True:
            ret, img_raw = video_capture.read()

            if ret:
                if idx == 0:
                    print('video shape : {}'.format(img_raw.shape))
                    # scale = 960/float(img_raw.shape[1])
                idx += 1
                if idx%2!=0:
                    continue
                # img = img_raw.copy()
                #------------------------------------------------------------------
                s_time = time.time()
                dets = detect_faces(ops,detect_model,img_raw)
                # for bbox in dets:
                #     cv2.rectangle(img_raw, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                get_faces_batch_landmarks(ops,dets,img_raw,use_cuda,draw_bbox = True)


                e_time = time.time()
                str_fps = ('{:.2f} Fps'.format(1./(e_time-s_time)))
                cv2.putText(img_raw, str_fps, (5,img_raw.shape[0]-3),cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 0, 255),4)
                cv2.putText(img_raw, str_fps, (5,img_raw.shape[0]-3),cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 0),1)
                print(str_fps)
                cv2.namedWindow('video',0)
                cv2.imshow('video',img_raw)
                if cv2.waitKey(1) == 27:
                    break
            else:
                break


    print('well done ~')
