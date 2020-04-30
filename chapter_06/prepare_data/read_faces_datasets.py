#-*-coding:utf-8-*-
# date:2020-04-05
# Author: xiang li
import os
import cv2 # 加载OpenCV库
import argparse # 加载处理命令行参数库
import shutil
import json
import numpy as np
import sys
sys.path.append('./')
from data_iter.data_agu import *
from utils.common_utils import *
import random

def refine_face_bbox(landmarks,img_shape):
    height,width = img_shape

    x = [float(pt[0]) for pt in landmarks]
    y = [float(pt[1]) for pt in landmarks]

    x1,y1,x2,y2 = np.min(x),np.min(y),np.max(x),np.max(y)

    expand_w = (x2-x1)
    expand_h = (y2-y1)

    x1 -= expand_w*0.1
    y1 -= expand_h*0.3
    x2 += expand_w*0.1
    y2 += expand_h*0.05

    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)

    x1 = int(max(0,x1))
    y1 = int(max(0,y1))
    x2 = int(min(x2,width-1))
    y2 = int(min(y2,height-1))

    return (x1,y1,x2,y2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--images_path', type=str, default = './datasets/WFLW_images/',
        help = 'images_path') # 图片路径
    parser.add_argument('--train_list', type=str,
        default = './datasets/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt',
        help = 'annotations_train_list')# 训练集标注信息

    parser.add_argument('--test_list', type=str,
        default = './datasets/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt',
        help = 'annotations_test_list')# 测试集标注信息
    parser.add_argument('--test_datasets', type=str,
        default = './datasets/test_expand_datasets/',
        help = 'test_datasets')# 测试集标注信息


    parser.add_argument('--img_size', type=tuple , default = (256,256),
        help = 'img_size') # 输入模型图片尺寸
    parser.add_argument('--fix_res', type=bool , default = False,
        help = 'fix_resolution') # 输入模型样本图片是否保证图像分辨率的长宽比

    parser.add_argument('--vis', type=bool, default = True,
        help = 'visualization')# 可视化标志位

    args = parser.parse_args()# 解析添加参数
    print('----------------------------------')

    unparsed = vars(args) # parse_args()方法的返回值为namespace，用vars()内建函数化为字典
    for key in unparsed.keys():
        print('{} : {}'.format(key,unparsed[key]))

    mkdir_(args.test_datasets)

    r_ = open(args.test_list,'r')
    lines = r_.readlines()

    idx = 0
    idx_save = 0
    for line in lines:
        # print(line)
        msg = line.strip().split(' ')
        idx += 1
        print('idx-',idx,' : ',len(msg))
        landmarks = msg[0:196]
        bbox = msg[196:200]
        attributes = msg[200:206]
        img_file = msg[206]

        img = cv2.imread(args.images_path+img_file)

        pts = []
        for i in range(int(len(landmarks)/2)):
            x = float(landmarks[i*2+0])
            y = float(landmarks[i*2+1])
            pts.append([x,y])

        refine_bbox = refine_face_bbox(pts,(img.shape[0],img.shape[1]))

        img_crop = img.copy()[refine_bbox[1]:refine_bbox[3],refine_bbox[0]:refine_bbox[2],:]





        #----------------------------------------------------------------------- save image
        tx1,ty1,tx2,ty2 = int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])

        expand_w = (tx2-tx1)
        expand_h = (ty2-ty1)

        tx1 -= expand_w*0.1
        ty1 -= expand_h*0.2
        tx2 += expand_w*0.1
        ty2 += expand_h*0.05

        tx1,ty1,tx2,ty2 = int(tx1),int(ty1),int(tx2),int(ty2)

        tx1 = int(max(0,tx1))
        ty1 = int(max(0,ty1))
        tx2 = int(min(tx2,img.shape[1]-1))
        ty2 = int(min(ty2,img.shape[0]-1))

        test_crop  = img[ty1:ty2,tx1:tx2,:]


        dict_test = {}
        dict_test['pts'] = []
        for i in range(int(len(landmarks)/2)):
            x = float(landmarks[i*2+0])
            y = float(landmarks[i*2+1])
            dict_test['pts'].append([x-tx1,y-ty1])
        img_name = img_file.split('/')[-1]
        idx_save += 1
        cv2.imwrite(args.test_datasets+'image_{}'.format(idx_save)+img_name,test_crop)

        fs = open(args.test_datasets+ 'image_{}'.format(idx_save)+img_name.replace('.jpg','.json'),"w",encoding='utf-8')
        json.dump(dict_test,fs,ensure_ascii=False,indent = 1,cls = JSON_Encoder)
        fs.close()

        #-----------------------------------------------------------------------

        left_eye = np.average(pts[60:68], axis=0)
        right_eye = np.average(pts[68:76], axis=0)
        # print(left_eye,right_eye)

        angle_random = random.randint(-25,25)
        # print(' angle_random : {}'.format(angle_random))
        img_rot, crop_pts  = face_random_rotate(img, pts, angle_random, left_eye, right_eye,
            fix_res = args.fix_res,img_size = args.img_size,vis = args.vis)

        #-----------------------------------
        if args.vis:

            cv2.circle(img, (int(left_eye[0]),int(left_eye[1])), 5, (255,0,255),-1)
            cv2.circle(img, (int(right_eye[0]),int(right_eye[1])), 5, (255,0,255),-1)

            cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (0,255,0), 4)
            cv2.rectangle(img, (int(refine_bbox[0]),int(refine_bbox[1])), (int(refine_bbox[2]),int(refine_bbox[3])), (0,155,255), 4)

            for i in range(int(len(landmarks)/2)):
                x = float(landmarks[i*2+0])
                y = float(landmarks[i*2+1])
                cv2.circle(img, (int(x),int(y)), 1, (255,0,0),-1)

            cv2.namedWindow('image',0)
            cv2.imshow('image',img)
            cv2.namedWindow('rotate',0)
            cv2.imshow('rotate',img_rot)
            cv2.namedWindow('face_crop',0)
            cv2.imshow('face_crop',img_crop)
            if cv2.waitKey(100) == 27:
                break

    if args.vis:
        cv2.destroyAllWindows()


    # unparsed['m'] = 999
    # fs = open('./make_data_train_test_msg.tr_param',"w",encoding='utf-8')
    # json.dump(unparsed,fs,ensure_ascii=False,indent = 1)
    # fs.close()
    #
    # mkdir_(args.train_datasets,flag_rm = True)# 创建训练集文件夹
    # mkdir_(args.test_datasets,flag_rm = True)# 创建测试集文件夹
    #
    # imgs_list = open(args.images_list,'r',encoding = 'utf-8').readlines()# 图片列表
    # print(len(imgs_list))
    # train_test_split_list = open(args.train_test_split,'r',encoding = 'utf-8').readlines()# 训练和测试集分割列表
    # print(len(train_test_split_list))
    # bbox_msg_list = open(args.bbox_msg,'r',encoding = 'utf-8').readlines()# 训练和测试集分割列表
    # print(len(train_test_split_list))
    #
    # train_cnt = 0
    # test_cnt = 0
    # img_mean_w = 0.# 统计crop图片 width 均值
    # img_mean_h = 0.# 统计crop图片 height 均值
    # for i in range(len(imgs_list)):
    #     image_id,image_name = imgs_list[i].strip('\n').split(' ')# 获取信息 ：图片id 、图片相对路径
    #     doc_class = image_name.split('/')[0]# 每一类文件夹
    #     _,is_train = train_test_split_list[i].strip('\n').split(' ')# 获取信息 ：图片id 、是否是训练图片标志位
    #     is_train = int(is_train)
    #     _,x,y,w,h = bbox_msg_list[i].strip('\n').split(' ')# 获取信息 ：图片id 、边界框
    #     x1,y1,x2,y2 = int(float(x)),int(float(y)),int(float(x) + float(w)),int(float(y) + float(h))
    #     print('[{}/{}] {} : is_train {} '.format(image_id,len(imgs_list),image_name,is_train))
    #
    #     if is_train:# 训练集
    #         path_s = args.train_datasets
    #         train_cnt += 1
    #     else:# 测试集
    #         path_s = args.test_datasets
    #         test_cnt += 1
    #
    #     mkdir_(path_s + doc_class)# 创建训练测试集的分类目录
    #
    #     if args.make_cropdata:# 是否制作 crop 数据集
    #         img = cv2.imread(args.images_path + image_name)
    #         cv2.imwrite(path_s +image_name,img[y1:y2,x1:x2,:])
    #         img_mean_w += (x2-x1)
    #         img_mean_h += (y2-y1)
    #         print('w,h : ({},{})'.format(int(img_mean_w/(train_cnt+test_cnt)),int(img_mean_h/(train_cnt+test_cnt))))
    #     else:# 制作原图数据集
    #         shutil.copyfile(args.images_path + image_name,path_s +image_name )# 将原图片路径拷贝到相应的训练集和测试集类别文件夹
    #
    #     if args.vis:
    #         img = cv2.imread(args.images_path + image_name)# 读取图片可视化
    #
    #         bbox = x1,y1,x2,y2
    #         plot_box(bbox, img, color=(255,0,0), label='bird')
    #
    #         cv2.namedWindow('img',0)#窗口大小可改变
    #         cv2.imshow('img',img)#显示旋转图片
    #         cv2.waitKey(1)
    #
    # print('train datasets len : {}'.format(train_cnt))
    # print('test datasets len : {}'.format(test_cnt))
    # if args.vis:
    #     cv2.destroyAllWindows()
