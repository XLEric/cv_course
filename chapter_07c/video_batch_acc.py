#coding:utf-8
# date:2019-08-08
# Author: X.li

import argparse
import time
import os
import sys
sys.path.append('./')
import torch
import torch.backends.cudnn as cudnn
from utils.datasets import *
from utils.utils import *
from utils.parse_config import parse_data_cfg
from yolov3 import Yolov3, Yolov3Tiny
from utils.torch_utils import select_device

from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from models.mobilenetv2 import MobileNetV2
from utils.common_utils import *
from acc_model import acc_model

def process_data(img, img_size=416):# 图像预处理
    img, _, _, _ = letterbox(img, height=img_size)
    # Normalize RGB
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    return img

def yolo_model_param(model):
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        print("该层的结构: {}, 参数和: {}".format(str(list(i.size())), str(l)))
        k = k + l
    print("----------------------")
    print("总参数数量和: {}".format(str(k)))

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

class yolo_detect(object):
    def __init__(self,ops,device):
        self.ops = ops
        self.img_size = ops.img_size
        self.classes = load_classes(parse_data_cfg(ops.data_cfg)['names'])
        self.num_classes = len(self.classes)

        if "tiny" in ops.detect_network:
            a_scalse = 416./ops.img_size
            anchors=[(10, 14), (23, 27), (37, 58), (81, 82), (135, 169), (344, 319)]
            anchors_new = [ (int(anchors[j][0]/a_scalse),int(anchors[j][1]/a_scalse)) for j in range(len(anchors)) ]
            model = Yolov3Tiny(self.num_classes,anchors = anchors_new)
            weights = ops.detect_model
            print('network : yolov3 - tiny')
        else:
            a_scalse = 416./ops.img_size
            anchors=[(10,13), (16,30), (33,23), (30,61), (62,45), (59,119), (116,90), (156,198), (373,326)]
            anchors_new = [ (int(anchors[j][0]/a_scalse),int(anchors[j][1]/a_scalse)) for j in range(len(anchors)) ]
            model = Yolov3(self.num_classes,anchors = anchors_new)
            weights = ops.detect_model
            print('network : yolov3')

        self.model = model
        yolo_model_param(self.model)# 显示模型参数

        self.device = device
        self.use_cuda = torch.cuda.is_available()
        # Load weights
        if os.access(weights,os.F_OK):# 判断模型文件是否存在
            self.model.load_state_dict(torch.load(weights, map_location=lambda storage, loc: storage)['model'])
            self.model.eval()
            print('\n/******************* Yolo V3 acc  ******************/')
            acc_model(ops,self.model)
            self.model = self.model.to(self.device)
        else:
            print('------- >>> error model not exists')
            return False

    def predict(self, img_):
        detections = None
        flag_dinner_dirty = False
        with torch.no_grad():
            img = process_data(img_, self.img_size)
            img = torch.from_numpy(img).unsqueeze(0).to(self.device)

            pred, _ = self.model(img)#图片检测

            detections = non_max_suppression(pred, self.ops.conf_thres, self.ops.nms_thres)[0] # nms

            if detections is None or len(detections) == 0:
                return []
            detections[:, :4] = scale_coords(self.img_size, detections[:, :4], img_.shape).round()

            # 绘制检测结果 ：detect reslut
            dets = []
            for *xyxy, conf, cls_conf, cls in detections:
                x1,y1,x2,y2 = xyxy
                dets.append([x1.item(),y1.item(),x2.item(),y2.item(),conf.item()])
                # print('[x1,y1,x2,y2,conf : ',x1,y1,x2,y2,conf)

        return detections
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=' YOLO V3 Inferece')
    parser.add_argument('-m', '--detect_model', default='./weights-yolov3-tiny/Detect115.pt',
                        type=str, help='Trained state_dict file path to open')

    parser.add_argument('--detect_network', type=str, default = 'tiny',help = 'tiny,yolo') # detect_network 选择

    parser.add_argument('--GPUS', type=str, default = '0',help = 'GPUS') # GPU选择

    parser.add_argument('--data_cfg', type=str, default = './cfg/voc_faces.data',help = 'data_cfg') #
    parser.add_argument('--img_size', type=int, default = 416, help = 'img_size') #
    parser.add_argument('--conf_thres', type=float, default = 0.35,help = 'conf_thres') #
    parser.add_argument('--vis_thres', type=float, default = 0.35,help = 'vis_thres') #
    parser.add_argument('--nms_thres', type=float, default = 0.3,help = 'nms_thres') #
    #-----------------------------------------------------------------------------------------
    parser.add_argument('--landmarks_model', type=str, default = './landmarks_model/resnet50_epoch-2350.pth',
        help = 'landmarks_model') # 模型路径
    parser.add_argument('--landmarks_network', type=str, default = 'resnet_50',
        help = 'model : resnet_18,resnet_34,resnet_50,resnet_101,resnet_152,mobilenetv2') # 模型类型
    parser.add_argument('--landmarks_num_classes', type=int , default = 196,
        help = 'landmarks_num_classes') #  分类类别个数
    parser.add_argument('--landmarks_img_size', type=tuple , default = (256,256),
        help = 'landmarks_img_size') # 输入landmarks 模型图片尺寸
    #-----------------------------------------------------------------------------------------
    parser.add_argument('--max_batch_size', type=int , default = 6,
        help = 'max_batch_size') #  最大 landmarks - max_batch_size
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

    device = torch.device("cuda:0" if use_cuda else "cpu")

    # 加载测试模型
    if os.access(ops.landmarks_model,os.F_OK):# checkpoint
        # chkpt = torch.load(ops.landmarks_model, map_location=device)
        # landmarks_model.load_state_dict(chkpt)
        print('load landmarks model : {}'.format(ops.landmarks_model))

        chkpt = torch.load(ops.landmarks_model, map_location=lambda storage, loc: storage)
        landmarks_model.load_state_dict(chkpt)
        landmarks_model.eval() # 设置为前向推断模式
        print('load landmarks model : {}'.format(ops.landmarks_model))
        print('\n/******************* landmarks model acc  ******************/')
        acc_model(ops,landmarks_model)
    landmarks_model = landmarks_model.to(device)


    #--------------------------------------------------------------------------- 构建人脸检测模型
    detect_model = yolo_detect(ops = ops,device = device)

    with torch.no_grad():#设置无梯度运行
        video_capture = cv2.VideoCapture(ops.test_path)

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

                    dets = detect_model.predict(img_raw)

                    if len(dets)> 0:
                        get_faces_batch_landmarks(ops,dets,img_raw,use_cuda,draw_bbox = True)

                    e_time = time.time()
                    str_fps = ('{:.2f} Fps'.format(1./(e_time-s_time)))
                    cv2.putText(img_raw, str_fps, (5,img_raw.shape[0]-3),cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 0, 255),4)
                    cv2.putText(img_raw, str_fps, (5,img_raw.shape[0]-3),cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 0),1)

                    cv2.namedWindow('video',0)
                    cv2.imshow('video',img_raw)
                    if cv2.waitKey(1) == 27:
                        break
                else:
                    break
        cv2.destroyAllWindows()
