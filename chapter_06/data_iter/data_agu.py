import cv2
import numpy as np
import random

# flip 的 landmarks 查表之前 landmarks 序号。
flip_landmarks_dict = {
    0:32,1:31,2:30,3:29,4:28,5:27,6:26,7:25,8:24,9:23,10:22,11:21,12:20,13:19,14:18,15:17,
    16:16,17:15,18:14,19:13,20:12,21:11,22:10,23:9,24:8,25:7,26:6,27:5,28:4,29:3,30:2,31:1,32:0,
    33:46,34:45,35:44,36:43,37:42,38:50,39:49,40:48,41:47,
    46:33,45:34,44:35,43:36,42:37,50:38,49:39,48:40,47:41,
    60:72,61:71,62:70,63:69,64:68,65:75,66:74,67:73,
    72:60,71:61,70:62,69:63,68:64,75:65,74:66,73:67,
    96:97,97:96,
    51:51,52:52,53:53,54:54,
    55:59,56:58,57:57,58:56,59:55,
    76:82,77:81,78:80,79:79,80:78,81:77,82:76,
    87:83,86:84,85:85,84:86,83:87,
    88:92,89:91,90:90,91:89,92:88,
    95:93,94:94,93:95
    }
# 非形变处理
def letterbox(img_,img_size=256,mean_rgb = (128,128,128)):

    shape_ = img_.shape[:2]  # shape = [height, width]
    ratio = float(img_size) / max(shape_)  # ratio  = old / new
    new_shape_ = (round(shape_[1] * ratio), round(shape_[0] * ratio))
    dw_ = (img_size - new_shape_[0]) / 2  # width padding
    dh_ = (img_size - new_shape_[1]) / 2  # height padding
    top_, bottom_ = round(dh_ - 0.1), round(dh_ + 0.1)
    left_, right_ = round(dw_ - 0.1), round(dw_ + 0.1)
    # resize img
    img_a = cv2.resize(img_, new_shape_, interpolation=cv2.INTER_LINEAR)

    img_a = cv2.copyMakeBorder(img_a, top_, bottom_, left_, right_, cv2.BORDER_CONSTANT, value=mean_rgb)  # padded square

    return img_a

# def scale_coords(img_size, coords, img0_shape):# image size 转为 原图尺寸
#     # Rescale x1, y1, x2, y2 from 416 to image size
#     # print('coords     : ',coords)
#     # print('img0_shape : ',img0_shape)
#     gain = float(img_size) / max(img0_shape)  # gain  = old / new
#     # print('gain       : ',gain)
#     pad_x = (img_size - img0_shape[1] * gain) / 2  # width padding
#     pad_y = (img_size - img0_shape[0] * gain) / 2  # height padding
#     # print('pad_xpad_y : ',pad_x,pad_y)
#     coords[:, [0, 2]] -= pad_x
#     coords[:, [1, 3]] -= pad_y
#     coords[:, :4] /= gain
#     coords[:, :4] = torch.clamp(coords[:, :4], min=0)# 夹紧区间最小值不为负数
#     return coords

def img_agu_channel_same(img_):
    img_a = np.zeros(img_.shape, dtype = np.uint8)
    gray = cv2.cvtColor(img_,cv2.COLOR_RGB2GRAY)
    img_a[:,:,0] =gray
    img_a[:,:,1] =gray
    img_a[:,:,2] =gray

    return img_a

# 图像旋转
def face_random_rotate(image , pts,angle ,Eye_Left,Eye_Right,fix_res= True,img_size=(256,256),vis = False):
    cx,cy = (Eye_Left[0] + Eye_Right[0]) / 2,(Eye_Left[1] + Eye_Right[1]) / 2
    (h , w) = image.shape[:2]
    h = h
    w = w
    # (cx , cy) = (int(0.5 * w) , int(0.5 * h))
    M = cv2.getRotationMatrix2D((cx , cy) , angle , 1.0)
    cos = np.abs(M[0 , 0])
    sin = np.abs(M[0 , 1])

    # 计算新图像的bounding
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0 , 2] += int(0.5 * nW) - cx
    M[1 , 2] += int(0.5 * nH) - cy


    resize_model = [cv2.INTER_LINEAR,cv2.INTER_CUBIC,cv2.INTER_NEAREST,cv2.INTER_AREA,cv2.INTER_LANCZOS4]

    img_rot = cv2.warpAffine(image , M , (nW , nH),flags=resize_model[random.randint(0,4)])
    #flags : INTER_LINEAR INTER_CUBIC INTER_NEAREST
    #borderMode : BORDER_REFLECT BORDER_TRANSPARENT BORDER_REPLICATE CV_BORDER_WRAP BORDER_CONSTANT

    pts_r = []
    for pt in pts:
        x = float(pt[0])
        y = float(pt[1])

        x_r = (x*M[0][0] + y*M[0][1] + M[0][2])
        y_r = (x*M[1][0] + y*M[1][1] + M[1][2])

        pts_r.append([x_r,y_r])

    x = [pt[0] for pt in pts_r]
    y = [pt[1] for pt in pts_r]

    x1,y1,x2,y2 = np.min(x),np.min(y),np.max(x),np.max(y)

    expand_w = (x2-x1)*0.15,max((x2-x1)*0.15,30)
    expand_h = (y2-y1)*0.15,max((y2-y1)*0.15,30)


    translation_pixels = 60

    scaling = 0.3
    x1 += random.randint(-int(max((x2-x1)*scaling,translation_pixels)),int((x2-x1)*0.15))
    y1 += random.randint(-int(max((y2-y1)*scaling,translation_pixels)),int((y2-y1)*0.15))
    x2 += random.randint(-int((x2-x1)*0.15),int(max((x2-x1)*scaling,translation_pixels)))
    y2 += random.randint(-int((y2-y1)*0.15),int(max((y2-y1)*scaling,translation_pixels)))

    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
    x1 = int(max(0,x1))
    y1 = int(max(0,y1))
    x2 = int(min(x2,img_rot.shape[1]-1))
    y2 = int(min(y2,img_rot.shape[0]-1))


    crop_rot = img_rot[y1:y2,x1:x2,:]

    crop_pts = []
    width_crop = float(x2-x1)
    height_crop = float(y2-y1)
    for pt in pts_r:
        x = pt[0]
        y = pt[1]
        crop_pts.append([float(x-x1)/width_crop,float(y-y1)/height_crop]) # 归一化

    # 随机镜像
    if random.randint(0,1) == 0:
        # print('--------->>> flip')
        crop_rot = cv2.flip(crop_rot,1)
        crop_pts_flip = []
        for i in range(len(crop_pts)):
            # print( crop_rot.shape[1],crop_pts[flip_landmarks_dict[i]][0])
            x = 1. - crop_pts[flip_landmarks_dict[i]][0]
            y = crop_pts[flip_landmarks_dict[i]][1]
            # print(i,x,y)
            crop_pts_flip.append([x,y])
        crop_pts = crop_pts_flip

    if vis:
        for pt in crop_pts:
            x = int(pt[0]*width_crop)
            y = int(pt[1]*height_crop)

            cv2.circle(crop_rot, (int(x),int(y)), 2, (255,0,255),-1)

    if fix_res:
        crop_rot = letterbox(crop_rot,img_size=img_size[0],mean_rgb = (128,128,128))
    else:
        crop_rot = cv2.resize(crop_rot, img_size, interpolation = resize_model[random.randint(0,4)])

    return crop_rot,crop_pts
