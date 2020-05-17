# -*- coding: utf-8 -*-

import random
import math
import threading
import time
import logging
import queue
import cv2
import numpy
from ChasingTrainFramework_GeneralOneClassDetection.image_augmentation.augmentor import Augmentor
from ChasingTrainFramework_GeneralOneClassDetection.data_iterator_base.data_batch import DataBatch

class Multithread_DataIter_for_CrossEntropy:

    def __init__(self,
                 torch_module,
                 num_threads,
                 data_provider,
                 batch_size,
                 enable_horizon_flip,
                 enable_vertical_flip,
                 enable_random_brightness,
                 brightness_params,
                 enable_random_saturation,
                 saturation_params,
                 enable_random_contrast,
                 contrast_params,
                 enable_blur,
                 blur_params,
                 blur_kernel_size_list,
                 neg_image_ratio,
                 num_image_channels,
                 net_input_height,
                 net_input_width,
                 num_output_scales,
                 receptive_field_list,
                 receptive_field_stride,
                 feature_map_size_list,
                 receptive_field_center_start,
                 bbox_small_list,
                 bbox_large_list,
                 bbox_small_gray_list,
                 bbox_large_gray_list,
                 num_output_channels,
                 neg_image_resize_factor_interval
                 ):

        self.torch_module = torch_module

        self.num_thread = num_threads
        logging.info('Prepare the data provider for all dataiter threads ---- ')

        self.data_provider = data_provider
        self.positive_index = self.data_provider.positive_index
        self.negative_index = self.data_provider.negative_index
        logging.info('Dataset statistics:\n\t%d positive images;\t%d negative images;\t%d images in total.',
                     len(self.positive_index), len(self.negative_index),
                     len(self.positive_index) + len(self.negative_index))

        # augmentation settings
        self.enable_horizon_flip = enable_horizon_flip
        self.enable_vertical_flip = enable_vertical_flip
        self.pixel_augmentor_func_list = []

        self.enable_random_brightness = enable_random_brightness
        self.brightness_params = brightness_params
        def brightness_augmentor(input_im):
            if self.enable_random_brightness and random.random() > 0.5:
                input_im = Augmentor.brightness(input_im, **self.brightness_params)
                return input_im
            else:
                return input_im
        self.pixel_augmentor_func_list.append(brightness_augmentor)

        self.enable_random_saturation = enable_random_saturation
        self.saturation_params = saturation_params
        def saturation_augmentor(input_im):
            if self.enable_random_saturation and random.random() > 0.5:
                input_im = Augmentor.saturation(input_im, **self.saturation_params)
                return input_im
            else:
                return input_im
        self.pixel_augmentor_func_list.append(saturation_augmentor)

        self.enable_random_contrast = enable_random_contrast
        self.contrast_params = contrast_params
        def contrast_augmentor(input_im):
            if self.enable_random_contrast and random.random() > 0.5:
                input_im = Augmentor.contrast(input_im, **self.contrast_params)
                return input_im
            else:
                return input_im
        self.pixel_augmentor_func_list.append(contrast_augmentor)

        self.enable_blur = enable_blur
        self.blur_params = blur_params
        self.blur_kernel_size_list = blur_kernel_size_list
        def blur_augmentor(input_im):
            if self.enable_blur and random.random() > 0.5:
                kernel_size = random.choice(self.blur_kernel_size_list)
                self.blur_params['kernel_size'] = kernel_size
                input_im = Augmentor.blur(input_im, **self.blur_params)
                return input_im
            else:
                return input_im
        self.pixel_augmentor_func_list.append(blur_augmentor)

        self.batch_size = batch_size
        self.neg_image_ratio = neg_image_ratio
        self.num_neg_images_per_batch = int(self.neg_image_ratio * self.batch_size) if len(self.negative_index) else 0

        self.num_image_channels = num_image_channels
        self.net_input_height = net_input_height
        self.net_input_width = net_input_width


        self.num_output_scales = num_output_scales
        self.receptive_field_list = receptive_field_list
        self.feature_map_size_list = feature_map_size_list
        self.receptive_field_stride = receptive_field_stride
        self.bbox_small_list = bbox_small_list
        self.bbox_large_list = bbox_large_list
        self.receptive_field_center_start = receptive_field_center_start
        self.normalization_constant = [i / 2.0 for i in self.receptive_field_list]
        self.bbox_small_gray_list = bbox_small_gray_list
        self.bbox_large_gray_list = bbox_large_gray_list
        self.num_output_channels = num_output_channels
        self.neg_image_resize_factor_interval = neg_image_resize_factor_interval

        # prepare threads
        self.batch_queue = queue.Queue(maxsize=self.num_thread + 1)
        self.stop_flag = False

        # the main procedure running in a thread
        def thread_func(self):
            while True:
                if self.stop_flag:
                    break
                try:
                    self.batch_queue.put(self.__prepare_batch())
                except Exception as e:
                    print(e)

        self._threads = [threading.Thread(target=thread_func, args=[self])
                         for i in range(self.num_thread)]

        for thread in self._threads:
            thread.setDaemon(True)
            thread.start()

    def __del__(self):
        self.stop_flag = True

        for thread in self._threads:
            thread.join(5)

    def next(self):
        return self.batch_queue.get()

    def __prepare_batch(self):
        im_batch = numpy.zeros((self.batch_size,
                                self.num_image_channels,
                                self.net_input_height,
                                self.net_input_width),
                               dtype=numpy.float32)

        label_batch_list = [numpy.zeros((self.batch_size,
                                         self.num_output_channels,# 6
                                         v,
                                         v),
                                        dtype=numpy.float32)
                            for v in self.feature_map_size_list]
        # for i in range(len(label_batch_list)):
        #     print('label_batch_list : ',i+1,')',label_batch_list[i].shape)

        mask_batch_list = [numpy.zeros((self.batch_size,
                                        self.num_output_channels,
                                        v,
                                        v),
                                       dtype=numpy.float32)
                           for v in self.feature_map_size_list]

        data_batch = DataBatch(self.torch_module)

        loop = 0
        while loop < self.batch_size:
            # 先获得足够的负样本，正负样本的比例为10:1
            if loop < self.num_neg_images_per_batch:  # fill neg images first
                # 随机选择一个负样本(图片中没有行人)下标
                rand_idx = random.choice(self.negative_index)
                # 通过负样本（图片中没有行人）下标获得图片
                im, _, __ = self.data_provider.read_by_index(rand_idx)
                # 获得一个书记缩放大小的因子random_resize_factor
                random_resize_factor = random.random() * (self.neg_image_resize_factor_interval[1] - self.neg_image_resize_factor_interval[0]) + self.neg_image_resize_factor_interval[0]
                # 把原图缩放到原来的random_resize_factor倍
                im = cv2.resize(im, (0, 0), fy=random_resize_factor, fx=random_resize_factor)

                # 输入图像的高框都减去640（默认配置）
                # print(' self.net_input_height,self.net_input_width : ',self.net_input_height,self.net_input_width)
                h_interval = im.shape[0] - self.net_input_height
                w_interval = im.shape[1] - self.net_input_width

                # 如果图片的高能够大于640（默认配置）大小
                if h_interval >= 0:
                    y_top = random.randint(0, h_interval)
                else:# 如果图片的长小于640，
                    y_pad = int(-h_interval / 2)# 在不足的地方使用0像素填充
                if w_interval >= 0:# 如果图片的宽大于640
                    x_left = random.randint(0, w_interval) # 随机在图片左侧选择一个x值坐标
                else:
                    x_pad = int(-w_interval / 2) # 要是图片的宽小于640，使用0进行填补
                # 输入图片初始化为0
                im_input = numpy.zeros((self.net_input_height, self.net_input_width, self.num_image_channels),
                                       dtype=numpy.uint8)
                # 对于该剪切的图片进行剪切，该填补的的像素使用默认0处理
                if h_interval >= 0 and w_interval >= 0:
                    im_input[:, :, :] = im[y_top:y_top + self.net_input_height, x_left:x_left + self.net_input_width, :]
                elif h_interval >= 0 and w_interval < 0:
                    im_input[:, x_pad:x_pad + im.shape[1], :] = im[y_top:y_top + self.net_input_height, :, :]
                elif h_interval < 0 and w_interval >= 0:
                    im_input[y_pad:y_pad + im.shape[0], :, :] = im[:, x_left:x_left + self.net_input_width, :]
                else:
                    im_input[y_pad:y_pad + im.shape[0], x_pad:x_pad + im.shape[1], :] = im[:, :, :]

                # data augmentation 数据增强，进行垂直或者水平翻转
                if self.enable_horizon_flip and random.random() > 0.5:
                    im_input = Augmentor.flip(im_input, 'h')
                if self.enable_vertical_flip and random.random() > 0.5:
                    im_input = Augmentor.flip(im_input, 'v')
                # 数据增强，对比度，模糊，亮度等等
                if random.random() > 0.5:
                    random.shuffle(self.pixel_augmentor_func_list)
                    for augmentor in self.pixel_augmentor_func_list:
                        im_input = augmentor(im_input)

                # display for debug-------------------------------------------------
                # cv2.imshow('im', im_pad.astype(dtype=numpy.uint8))
                # cv2.waitKey()

                # cv2.namedWindow('dataIter',0)
                # cv2.imshow('dataIter',im_input)
                # cv2.waitKey(0)

                # 根据 PyTorch 要求进行数据类型 及通道转换，
                im_input = im_input.astype(numpy.float32)
                im_input = im_input.transpose([2, 0, 1])
                # 以上是对一张图片的处理，把获得的图像加载到im_batch
                im_batch[loop] = im_input
                for label_batch in label_batch_list:
                    # 标记为负样本，box以及后面
                    # 输出的的6个通道，第1个通道标记为1，其余为0
                    label_batch[loop, 1, :, :] = 1
                for mask_batch in mask_batch_list:
                    # 输出的6个通道，第0和第1个标记为1，其余为0
                    mask_batch[loop, 0:2, :, :] = 1

            else:
                # 随机获得一张正样本的图片，以及初始的bboxes
                rand_idx = random.choice(self.positive_index)
                im, _, bboxes_org = self.data_provider.read_by_index(rand_idx)
                # 获得boxes的个数，并且进行拷贝
                num_bboxes = bboxes_org.shape[0]

                bboxes = bboxes_org.copy()

                # data augmentation ,如果进行了水平或者或者垂直翻转，则对应的boxes也要进行翻转
                if self.enable_horizon_flip and random.random() > 0.5:
                    im = Augmentor.flip(im, 'h')
                    bboxes[:, 0] = im.shape[1] - (bboxes[:, 0] + bboxes[:, 2])
                if self.enable_vertical_flip and random.random() > 0.5:
                    im = Augmentor.flip(im, 'v')
                    bboxes[:, 1] = im.shape[0] - (bboxes[:, 1] + bboxes[:, 3])

                # display for debug-------------------------------------------
                # im_show = im.copy()
                # for n in range(num_bboxes):
                #     cv2.rectangle(im_show, (int(bboxes[n,0]),int(bboxes[n,1])), (int(bboxes[n,0]+bboxes[n,2]),int(bboxes[n,1]+bboxes[n,3])), (255,255,0), 1)
                # cv2.imshow('im_show', im_show)
                # cv2.waitKey()

                # randomly select a bbox,随机选择一个boxes，这个是为了保证，后续在进行随机剪切的时候，图片中至少存在一个box
                # 确保其为一个正样本
                bbox_idx = random.randint(0, num_bboxes - 1)

                # randomly select a reasonable scale for the selected bbox (selection strategy may vary from task to task)
                # 根据boxes的尺寸大小，选择一个合理的缩放
                target_bbox = bboxes[bbox_idx, :]
                # 获得随机选择boxes的长度
                longer_side = max(target_bbox[2:])
                if longer_side <= self.bbox_small_list[0]:# bbox_small_list : [10, 20, 40, 80, 160]
                    scale_idx = 0
                elif longer_side <= self.bbox_small_list[1]:
                    scale_idx = random.randint(0, 1)
                # elif longer_side <= self.bbox_small_list[2]:
                #     scale_idx = random.randint(0, 2)
                else:
                    if random.random() > 0.9:
                        scale_idx = random.randint(0, self.num_output_scales)
                    else:
                        scale_idx = random.randint(0, self.num_output_scales - 1)

                # choose a side length in the selected scale
                # 保证side_length在选择的尺度之间
                if scale_idx == self.num_output_scales:
                    scale_idx -= 1
                    side_length = self.bbox_large_list[-1] + random.randint(0, self.bbox_large_list[-1] * 0.5)
                else:
                    side_length = self.bbox_small_list[scale_idx] + \
                                  random.randint(0, self.bbox_large_list[scale_idx] - self.bbox_small_list[scale_idx])
                # 获得缩放比例
                target_scale = float(side_length) / longer_side

                # resize bboxes
                # resize bboxes，对bboxes中的所有box以及target_bbox进行缩放。
                bboxes = bboxes * target_scale
                target_bbox = target_bbox * target_scale

                # determine the states of a bbox in each scale
                # 确定每个尺寸的box对应的green（论文eRF区域），gray（RF-eRF，注意不全是）区域，valid[RF]
                green = [[False for i in range(num_bboxes)] for j in range(self.num_output_scales)]
                gray = [[False for i in range(num_bboxes)] for j in range(self.num_output_scales)]
                valid = [[False for i in range(num_bboxes)] for j in range(self.num_output_scales)]

                # 对每个box进行处理，num_output_scales默认为6
                for i in range(num_bboxes):
                    temp_bbox = bboxes[i, :]
                    large_side = max(temp_bbox[2:])
                    for j in range(self.num_output_scales):
                        # 对每个缩放的尺寸的特征图进行处理 ，判断中心点落在那个尺寸图对应的green，gray，valid
                        if self.bbox_small_list[j] <= large_side <= self.bbox_large_list[j]:
                            green[j][i] = True
                            valid[j][i] = True
                        elif self.bbox_small_gray_list[j] <= large_side <= self.bbox_large_gray_list[j]:
                            gray[j][i] = True
                            valid[j][i] = True

                # resize the original image 把图片按照计算出来的缩放因子进行缩放
                im = cv2.resize(im, None, fx=target_scale, fy=target_scale)

                # crop the original image centered on the center of the selected bbox with vibration
                # 绕着target_bbox的中心进行剪切，并且保证中心在特征图之内
                vibration_length = int(self.receptive_field_stride[scale_idx] / 2)
                # 随机设置偏离box中心的偏移量
                offset_x = random.randint(-vibration_length, vibration_length)
                offset_y = random.randint(-vibration_length, vibration_length)
                crop_left = int(target_bbox[0] + target_bbox[2] / 2 + offset_x - self.net_input_width / 2.0)
                if crop_left < 0:
                    crop_left_pad = -int(crop_left)
                    crop_left = 0
                else:
                    crop_left_pad = 0
                crop_top = int(target_bbox[1] + target_bbox[3] / 2 + offset_y - self.net_input_height / 2.0)
                if crop_top < 0:
                    crop_top_pad = -int(crop_top)
                    crop_top = 0
                else:
                    crop_top_pad = 0
                crop_right = int(target_bbox[0] + target_bbox[2] / 2 + offset_x + self.net_input_width / 2.0)
                if crop_right > im.shape[1]:
                    crop_right = im.shape[1]

                crop_bottom = int(target_bbox[1] + target_bbox[3] / 2 + offset_y + self.net_input_height / 2.0)
                if crop_bottom > im.shape[0]:
                    crop_bottom = im.shape[0]

                im = im[crop_top:crop_bottom, crop_left:crop_right, :]
                im_input = numpy.zeros((self.net_input_height, self.net_input_width, 3), dtype=numpy.uint8)
                im_input[crop_top_pad:crop_top_pad + im.shape[0], crop_left_pad:crop_left_pad + im.shape[1], :] = im

                # image augmentation 数据增强
                if random.random() > 0.5:
                    random.shuffle(self.pixel_augmentor_func_list)
                    for augmentor in self.pixel_augmentor_func_list:
                        im_input = augmentor(im_input)

                # cv2.namedWindow('dataIter',0)
                # cv2.imshow('dataIter',im_input)
                # cv2.waitKey(0)

                # display for debug-------------------------------------------------
                # im_show = im_input.copy()
                # for n in range(num_bboxes):
                #     cv2.rectangle(im_show, (int(bboxes[n, 0] - crop_left + crop_left_pad), int(bboxes[n, 1] - crop_top + crop_top_pad)),
                #                   (int(bboxes[n, 0] + bboxes[n, 2] - crop_left + crop_left_pad),int(bboxes[n, 1] + bboxes[n, 3] - crop_top + crop_top_pad)),
                #                   (255, 0, 255), 1)
                # cv2.namedWindow('im_show',0)
                # cv2.imshow('im_show', im_show)
                # cv2.waitKey()
                #
                # print('---------------------------------------------------------------------->>>> im_input .shape',im_input.shape)

                im_input = im_input.astype(dtype=numpy.float32)
                im_input = im_input.transpose([2, 0, 1])

                # construct GT feature maps for each scale ，为每个尺寸的特征图构建map
                label_list = []
                mask_list = []

                # print('self.num_output_scales : ',self.num_output_scales)# 5
                # print('self.net_input_width height : ',self.net_input_width,self.net_input_height)# 640,640
                # print('self.num_output_channels : ',self.num_output_channels)# 6

                 # 分别对每个尺寸的feature maps进行处理
                for i in range(self.num_output_scales):# 5
                    # receptive_field_stride = [4, 8, 16, 32, 64]
                    # feature_map_size_list = [159, 79, 39, 19, 9]
                    # receptive_field_center_start = [3, 7, 15, 31, 63]
                    # compute the center coordinates of all RFs
                    # 计算所有RFs对应的中心点
                    receptive_field_centers = numpy.array(
                        [self.receptive_field_center_start[i] + w * self.receptive_field_stride[i] for w in range(self.feature_map_size_list[i])])
                    # print('receptive_field_centers : ',receptive_field_centers)# example : receptive_field_centers :  [ 63 127 191 255 319 383 447 511 575]
                    # print(i,') receptive_field_centers shape',receptive_field_centers.shape)# example : 9

                    # 求出box与剪切之后图片中心的偏移值
                    shift_x = (self.net_input_width / 2.0 - target_bbox[2] / 2) - target_bbox[0] - offset_x
                    shift_y = (self.net_input_height / 2.0 - target_bbox[3] / 2) - target_bbox[1] - offset_y
                    temp_label = numpy.zeros((self.num_output_channels, self.feature_map_size_list[i], self.feature_map_size_list[i]),
                                             dtype=numpy.float32)
                    temp_mask = numpy.zeros((self.num_output_channels, self.feature_map_size_list[i], self.feature_map_size_list[i]),
                                            dtype=numpy.float32)
                    temp_label[1, :, :] = 1
                    temp_mask[0:2, :, :] = 1
                    # 用来保存计算出来特征图对应的eRF区域为人脸的概率值
                    score_map_green = numpy.zeros((self.feature_map_size_list[i], self.feature_map_size_list[i]),
                                                  dtype=numpy.int32)
                    # 用来保存特征图的gray=RF-eRF区域为人脸的概率值
                    score_map_gray = numpy.zeros((self.feature_map_size_list[i], self.feature_map_size_list[i]),
                                                 dtype=numpy.int32)
                    for j in range(num_bboxes):
                        # 找到合适尺寸的，该预测box中心点的落在该尺寸的特征图上
                        if not valid[i][j]:
                            continue
                        # 获得该 bbox
                        temp_bbox = bboxes[j, :]

                        # skip the bbox that does not appear in the cropped area
                        # 如果box已经不在剪切之后图片的区域，则跳过对该box的处理
                        if temp_bbox[0] + temp_bbox[2] + shift_x <= 0 or temp_bbox[0] + shift_x >= self.net_input_width \
                                or temp_bbox[1] + temp_bbox[3] + shift_y <= 0 or temp_bbox[1] + shift_y >= self.net_input_height:
                            continue
                        # box最左边，最右边，最上边，最下边
                        temp_bbox_left_bound = temp_bbox[0] + shift_x
                        temp_bbox_right_bound = temp_bbox[0] + temp_bbox[2] + shift_x
                        temp_bbox_top_bound = temp_bbox[1] + shift_y
                        temp_bbox_bottom_bound = temp_bbox[1] + temp_bbox[3] + shift_y
                        # 把原图中的box映射到特征图上，得到特征图上的box，也就是RF对应在特征图像素的下标
                        left_RF_center_index = max(0, math.ceil((temp_bbox_left_bound - self.receptive_field_center_start[i]) / self.receptive_field_stride[i]))
                        right_RF_center_index = min(self.feature_map_size_list[i] - 1, math.floor((temp_bbox_right_bound - self.receptive_field_center_start[i]) / self.receptive_field_stride[i]))
                        top_RF_center_index = max(0, math.ceil((temp_bbox_top_bound - self.receptive_field_center_start[i]) / self.receptive_field_stride[i]))
                        bottom_RF_center_index = min(self.feature_map_size_list[i] - 1, math.floor((temp_bbox_bottom_bound - self.receptive_field_center_start[i]) / self.receptive_field_stride[i]))
                        # print('left_RF_center_index,right_RF_center_index,top_RF_center_index,bottom_RF_center_index: ',left_RF_center_index,right_RF_center_index,top_RF_center_index,bottom_RF_center_index)
                        # ignore the face with no RF centers inside 忽略掉没有RF中心的脸
                        if right_RF_center_index < left_RF_center_index or bottom_RF_center_index < top_RF_center_index:
                            continue
                        # 如果存在灰度区域，则存在gray区域，即RF-eRF。对gray进行赋值，全部为1
                        if gray[i][j]:
                            score_map_gray[top_RF_center_index:bottom_RF_center_index + 1, left_RF_center_index:right_RF_center_index + 1] = 1
                        # 如果不在灰色区域，原图中的box映射到box的eRF上面
                        else:
                            score_map_green[top_RF_center_index:bottom_RF_center_index + 1, left_RF_center_index:right_RF_center_index + 1] += 1
                            # 根据下标获得中心，并且进行了正则化
                            x_centers = receptive_field_centers[left_RF_center_index:right_RF_center_index + 1]
                            y_centers = receptive_field_centers[top_RF_center_index:bottom_RF_center_index + 1]
                            # 这里是机器要学习的东西，不是坐标点，而是距离中心坐标的的偏移量
                            x0_location_regression = (x_centers - temp_bbox_left_bound) / self.normalization_constant[i]
                            y0_location_regression = (y_centers - temp_bbox_top_bound) / self.normalization_constant[i]
                            x1_location_regression = (x_centers - temp_bbox_right_bound) / self.normalization_constant[i]
                            y1_location_regression = (y_centers - temp_bbox_bottom_bound) / self.normalization_constant[i]

                            # print('   x_centers:{}, y_centers:{}, x0:{}, y0:{}, x1:{}, y1:{} '.format(x_centers,y_centers,x0_location_regression,y0_location_regression,x1_location_regression,y1_location_regression))
                             # 对temp_label进行复制，temp_label是特征图的大小
                            temp_label[2, top_RF_center_index:bottom_RF_center_index + 1,
                            left_RF_center_index:right_RF_center_index + 1] = \
                                numpy.tile(x0_location_regression, [bottom_RF_center_index - top_RF_center_index + 1, 1])

                            # print('temp_label 2 : ',temp_label[2, top_RF_center_index:bottom_RF_center_index + 1,left_RF_center_index:right_RF_center_index + 1])

                            temp_label[3, top_RF_center_index:bottom_RF_center_index + 1,
                            left_RF_center_index:right_RF_center_index + 1] = \
                                numpy.tile(y0_location_regression, [right_RF_center_index - left_RF_center_index + 1, 1]).T

                            temp_label[4, top_RF_center_index:bottom_RF_center_index + 1,
                            left_RF_center_index:right_RF_center_index + 1] = \
                                numpy.tile(x1_location_regression, [bottom_RF_center_index - top_RF_center_index + 1, 1])

                            temp_label[5, top_RF_center_index:bottom_RF_center_index + 1,
                            left_RF_center_index:right_RF_center_index + 1] = \
                                numpy.tile(y1_location_regression, [right_RF_center_index - left_RF_center_index + 1, 1]).T
                    # 举例： score_gray_flag[59,59], 对gray区域进行标记，离开eRF=green越远，score越低
                    score_gray_flag = numpy.logical_or(score_map_green > 1, score_map_gray > 0)
                    location_green_flag = score_map_green == 1
                    # 标记为这是一个正样本
                    temp_label[0, :, :][location_green_flag] = 1
                    temp_label[1, :, :][location_green_flag] = 0
                     # 对第0个和第一个通道，不进行mask操作，因为其是用来标记正负样本的，其余的通道可以看作同样的mask
                    for c in range(self.num_output_channels):
                        if c == 0 or c == 1:
                            temp_mask[c, :, :][score_gray_flag] = 0
                            continue
                        # for bbox regression, only green area is available
                        temp_mask[c, :, :][location_green_flag] = 1

                    # display for debug----------------------------------------------------------------
                    # temp_label_score_show = temp_label[0, :, :] * temp_mask[0, :, :]
                    # temp_label_score_show = temp_label_score_show * 255
                    # cv2.imshow('temp_label_score_show',0)
                    # cv2.imshow('temp_label_score_show', cv2.resize(temp_label_score_show.astype(dtype=numpy.uint8), (0, 0), fx=2, fy=2))
                    # cv2.waitKey()

                    label_list.append(temp_label)
                    mask_list.append(temp_mask)

                im_batch[loop] = im_input
                for n in range(self.num_output_scales):
                    label_batch_list[n][loop] = label_list[n]
                    mask_batch_list[n][loop] = mask_list[n]
            loop += 1

        data_batch.append_data(im_batch)

        for n in range(self.num_output_scales):
            data_batch.append_label(mask_batch_list[n])
            data_batch.append_label(label_batch_list[n])

        return data_batch

    def get_batch_size(self):
        return self.batch_size


if __name__ == '__main__':
    import sys
    sys.path.append('..')
    from config_farm import configuration_10_320_20L_5scales_v2 as cfg
    from data_provider_farm.pickle_provider import PickleProvider
    import torch

    train_data_provider = PickleProvider(cfg.param_trainset_pickle_file_path)
    train_dataiter = Multithread_DataIter_for_CrossEntropy(
        torch_module=torch,
        num_threads=cfg.param_num_thread_train_dataiter,
        data_provider=train_data_provider,
        batch_size=cfg.param_train_batch_size,
        enable_horizon_flip=cfg.param_enable_horizon_flip,
        enable_vertical_flip=cfg.param_enable_vertical_flip,
        enable_random_brightness=cfg.param_enable_random_brightness,
        brightness_params=cfg.param_brightness_factors,
        enable_random_saturation=cfg.param_enable_random_saturation,
        saturation_params=cfg.param_saturation_factors,
        enable_random_contrast=cfg.param_enable_random_contrast,
        contrast_params=cfg.param_contrast_factors,
        enable_blur=cfg.param_enable_blur,
        blur_params=cfg.param_blur_factors,
        blur_kernel_size_list=cfg.param_blur_kernel_size_list,
        neg_image_ratio=cfg.param_neg_image_ratio,
        num_image_channels=cfg.param_num_image_channel,
        net_input_height=cfg.param_net_input_height,
        net_input_width=cfg.param_net_input_width,
        num_output_scales=cfg.param_num_output_scales,
        receptive_field_list=cfg.param_receptive_field_list,
        receptive_field_stride=cfg.param_receptive_field_stride,
        feature_map_size_list=cfg.param_feature_map_size_list,
        receptive_field_center_start=cfg.param_receptive_field_center_start,
        bbox_small_list=cfg.param_bbox_small_list,
        bbox_large_list=cfg.param_bbox_large_list,
        bbox_small_gray_list=cfg.param_bbox_small_gray_list,
        bbox_large_gray_list=cfg.param_bbox_large_gray_list,
        num_output_channels=cfg.param_num_output_channels,
        neg_image_resize_factor_interval=cfg.param_neg_image_resize_factor_interval
    )

    num_fetches = 200
    start = time.time()
    for i in range(num_fetches):
        batch = train_dataiter.next()
        print(i)
    time_elapsed = time.time() - start
    print('tatol fetching time: %f s; Speed: %.02f images/s' % (time_elapsed, num_fetches * cfg.param_train_batch_size / time_elapsed))
