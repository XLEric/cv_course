#-*-coding:utf-8-*-
# date:2019-06-18
# Author: xiang li
# function: detect data iterator

import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data_agu_fun import *
import xml.etree.cElementTree as ET
from data import preproc
flag_debug = False

random.seed(6)

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def get_datasets_label(path_image_):
    print('path_image : ',path_image_)
    all_label_ = {}
    for i,file in enumerate(os.listdir(path_image_)):

        if '.jpg' in file or '.png' in file:
            print(file)
            img_ = path_image_ + file
            # xml_ = img_.strip('.jpg').strip('.png')+'.xml'
            xml_ = img_.replace('.jpg','.xml')
            print(xml_)
            if not os.path.exists(xml_):
                continue
            list_x = get_xml_msg(xml_)
            # print(list_x)
            for j in range(len(list_x)):
                label_,bbox_ = list_x[j]

                if label_ not in all_label_.keys():
                    all_label_[label_] = 1
                else:
                    all_label_[label_] += 1

    label_num_ = {}
    class_ = 1
    for key_ in all_label_.keys():
        label_num_[key_] = class_
        class_ += 1

    return label_num_
def get_xml_msg(path):
    list_x = []
    tree=ET.parse(path)
    root=tree.getroot()
    for Object in root.findall('object'):
        name=Object.find('name').text
        #----------------------------
        # print('2) name',name)
        #----------------------------
        bndbox=Object.find('bndbox')
        xmin= np.float32((bndbox.find('xmin').text))
        ymin= np.float32((bndbox.find('ymin').text))
        xmax= np.float32((bndbox.find('xmax').text))
        ymax= np.float32((bndbox.find('ymax').text))
        bbox = int(xmin),int(ymin),int(xmax),int(ymax)
        xyxy = xmin,ymin,xmax,ymax
        list_x.append((name,xyxy))
    return list_x

class LoadImagesAndLabels(Dataset):  # for training/testing
    def __init__(self, path, img_size=(300,300),rgb_means=(104, 117, 123), is_train = False,augment=False):

        dict_label_ = get_datasets_label(path)
        print('dict_label : ',dict_label_)

        print('  img_size (height,width) : ',img_size[0],img_size[1])
        labels_ = []
        images_ = []

        for i,file_ in enumerate(os.listdir(path)):
            if '.jpg' in file_  or '.png' in file_:
                img_ = path + file_
                xml_ = img_.replace('.jpg','.xml')

                if not os.path.exists(xml_):
                    continue
                list_x = get_xml_msg(xml_)
                if len(list_x)>0:
                    print(i,') ',xml_)
                    images_.append(img_)
                    labels_.append(xml_)

        print('\n')
        self.label_num_dict = dict_label_
        self.labels = labels_
        self.files = images_
        self.img_size = img_size
        self.augment = augment
        self.is_train = is_train
        self.rgb_means = rgb_means
        self.preproc = preproc(img_size[0], rgb_means, 0.6)
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index]
        label_file_ = self.labels[index]
        img_ = cv2.imread(img_path)  # BGR
        list_x = get_xml_msg(label_file_)

        target_ = []
        for j in range(len(list_x)):
            label_,bbox_ = list_x[j]
            x1,y1,x2,y2 = bbox_[0],bbox_[1],bbox_[2],bbox_[3]
            target_.append((x1,y1,x2,y2,int(self.label_num_dict[label_])))

        if self.augment == True:
            if random.randint(0,16)==8:
                c = float(random.randint(50,150))/100.
                b = random.randint(-30,30)
                img_ = img_agu_contrast(img_, c, b)

            if random.randint(0,9)==1:
                img_ = img_agu_hue(img_)

            if random.randint(0,3)==1:
                img_ = img_agu_random_color_channel(img_)


        target_ = np.array(target_)


        img_, target_ = self.preproc(img_, target_)



        return img_, torch.from_numpy(target_)



"""Custom collate fn for dealing with batches of images that have a different
number of associated object annotations (bounding boxes).

Arguments:
    batch: (tuple) A tuple of tensor images and lists of annotations

Return:
    A tuple containing:
        1) (list of tensors) batch of images stacked on their 0 dim
        2) (list of tensors) annotations for a given image are stacked on 0 dim
"""
def detection_collate(batch):

    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(sample[1])
    return imgs, targets
