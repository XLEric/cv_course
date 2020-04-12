#-*-coding:utf-8-*-
# date:2020-03-28
# Author: xiang li
# function: read video

import cv2
if __name__ == "__main__":
    #读取 mp4 格式视频
    video_capture = cv2.VideoCapture('./video/a.mp4')
    while True:
        ret, img = video_capture.read()# 获取视频每一帧图像
        if ret == True:# 如果 ret 返回值为 True，显示图片
            cv2.namedWindow('frame',0)
            cv2.imshow("frame", img)
            key = cv2.waitKey(30)
            if key == 27:#当按键esc，退出显示
                break
        else:# 视频结尾 ret 返回 False，退出循环
            break
    video_capture.release()#释放视频
    cv2.destroyAllWindows()#关闭显示窗口
