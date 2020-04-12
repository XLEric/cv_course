import cv2 # 加载OpenCV库
import numpy as np # 加载Numpy库
if __name__ == "__main__":
    img = cv2.imread('000000003671.jpg')# 读取图片
    angle = 45 # 旋转角度，设为45度
    cx ,cy = int(img.shape[1]/2),int(img.shape[0]/2)# 定义旋转中心
    borderValue = 0 # 旋转空缺部分的缺省值，设为0
    (h , w) = img.shape[:2] # 图像的高和宽
    #计算旋转矩阵
    M = cv2.getRotationMatrix2D((cx , cy) , -angle , 1.0)
    cos = np.abs(M[0 , 0])
    sin = np.abs(M[0 , 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0 , 2] += int(0.5 * nW) - cx
    M[1 , 2] += int(0.5 * nH) - cy
    #更新旋转图片
    img_rot = cv2.warpAffine(img , M , (nW , nH),borderValue=borderValue)
    cv2.imshow('image',img)#显示原图
    cv2.namedWindow('img_rot',0)#窗口大小可改变
    cv2.imshow('img_rot',img_rot)#显示旋转图片
    cv2.waitKey(0)


    # img = cv2.imread('000000003671.jpg')
    # print('',img.shape)
    # a = 1.25 # 对比度
    # b = 20 # 亮度
    # img_ = img*a + b
    # img_[img_>255] = 255
    # img_[img_<0] = 0
    # img_= img_.astype(np.uint8)
    # cv2.imshow('image',img)#显示原图
    # cv2.imshow('image_',img_)#显示改变对比度和亮度的图片
    # cv2.waitKey(0)


    # img = cv2.imread('000000003671.jpg')
    # print('',img.shape)
    # x1,y1 = 100,150 # 裁剪部分的左上坐标
    # x2,y2 = 550,400 # 裁剪部分的右下坐标
    # img_crop = img[y1:y2,x1:x2,:]# 裁剪图片
    # cv2.imshow('image',img)#显示原图
    # cv2.namedWindow('crop',0)#窗口大小可改变
    # cv2.imshow('crop',img_crop)#显示裁剪后的图片
    # cv2.waitKey(0)


    # img_rgb = img.transpose(2, 0, 1)# BGR 转为 RGB
    # img_flip = cv2.flip(img,0)
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # cv2.imshow('image',img)
    # cv2.imshow('img_flip',img_flip)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # video_capture = cv2.VideoCapture('./NBA.mp4')# 读取 mp4 格式视频
    # while True:
    #     ret, img = video_capture.read()# 获取视频每一帧图像
    #     if ret == True:# 如果 ret 返回值为 True，显示图片
    #         cv2.namedWindow('frame',0)
    #         cv2.imshow("frame", img)
    #         key = cv2.waitKey(1)
    #         if key == 27:#当按键esc，退出显示
    #             break
    #     else:# 视频结尾 ret 返回 False，退出循环
    #         break
    # video_capture.release()#释放视频
    # cv2.destroyAllWindows()#关闭显示窗口

# if __name__ == "__main__":
#     img = cv2.imread('000000003671.jpg')
#     cv2.imshow('image',img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
