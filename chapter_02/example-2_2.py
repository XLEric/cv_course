#-*-coding:utf-8-*-
# date:2020-03-28
# Author: xiang li
# function: read camera

import cv2
if __name__ == "__main__":
    #读取摄像机
    camera_capture = cv2.VideoCapture(0)#0为相机ID
    while True:
        ret, img = camera_capture.read()# 获取相机图像
        if ret == True:# 如果 ret 返回值为 True，显示图片
            cv2.namedWindow('frame',0)
            cv2.imshow("frame", img)
            key = cv2.waitKey(1)
            if key == 27:#当按键esc，退出显示
                break
        else:# 相机读取失败，返回 False，退出循环
            break
    camera_capture.release()#释放相机
    cv2.destroyAllWindows()#关闭显示窗口
