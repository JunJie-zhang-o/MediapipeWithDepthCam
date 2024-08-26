#!/usr/bin/env python3
# coding=utf-8
'''
Author       : jay jay.zhangjunjie@outlook.com
Date         : 2024-08-01 13:44:28
LastEditTime: 2024-08-05 23:58:13
LastEditors: jay jay.zhangjunjie@outlook.com
Description  : 通过realsense和mediapipe,摇操作控制机器人,并结合触觉传感器实现自适应抓取
'''






from threading import Thread
import time
import cv2
import numpy as np

from utils.finger_angle import FingerAnglesHandle, handWorldLandmarks2List, drawFingerAngleOnImage
from utils.fps import FPS
from utils.realsense import RealSense
from xmlrpc import client
from utils.landmark import GestureLandMarkDetector, HandLandMarkDetector ,PoseLandMarkDetector
from utils.observer import GestureObserver
from utils.image_list import ImageList
from utils.body_handle import BodyObserver
import os,sys
from orbbec.gemini2 import Gemini2

os.environ["MEDIAPIPE_GPU_ENABLED"] = "1"










def showHandAngle(gestureDector:GestureLandMarkDetector, handle:FingerAnglesHandle):
    while 1:
        time.sleep(0.001)
        tt = time.time()
        if gestureDector.result is not None:
            handle.updata(handWorldLandmarks2List(gestureDector.result.hand_world_landmarks.toList()))
            # drawFingerAngleOnImage(gestureMarker.output_image, fAHandle.drawFingerAngleDatas)
            handle.drawAllFingerAngleOnImage(gestureDector.output_image, gestureDector.result.hand_landmarks.getAllJointPoint())
            print(f"Draw:{time.time() - tt}")



if __name__ == "__main__":



    gemini2 = Gemini2()
    gemini2.set_align_mode()

    # handMarker = HandLandMark(model_path="model/hand_landmarker.task")
    poseMarker = PoseLandMarkDetector(model_path="model/pose_landmarker_heavy.task",
    # poseMarker = PoseLandMarkDetector(model_path="model/pose_landmarker_full.task",
    # poseMarker = PoseLandMarkDetector(model_path="model/pose_landmarker_lite.task",
                                      min_pose_detection_confidence=0.5, # 0.7
                                      min_pose_presence_confidence=0.9,
                                      min_tracking_confidence=0.7)
    gestureMarker = GestureLandMarkDetector(model_path="model/gesture_recognizer.task")
                                            # min_hand_detection_confidence=0.3,
                                            # min_hand_presence_confidence=0.3,
                                            # min_tracking_confidence=0.5)

    gestureObs = GestureObserver(cam=gemini2)
    imageDeque = ImageList()
    # bodyObs = BodyObserver(imageDeque, gemini2)

    # gestureObs.register_callback(GestureObserver.FuncNameLists.INCREASE, lambda: print("INC") if bodyObs._cali_flag == True else None, duration=2, volatuationData=0.05)
    # gestureObs.register_callback(GestureObserver.FuncNameLists.REDUCE, lambda:print("DEC") if bodyObs._cali_flag == True else None, duration=2, volatuationData=0.05)

    gestureObs.register_callback(GestureObserver.FuncNameLists.INCREASE, lambda: client.ServerProxy("http://127.0.0.1:9120/", allow_none=True).open_no_block() if gestureObs._cali_flag == True else None, duration=1.5, volatuationData=0.06)
    gestureObs.register_callback(GestureObserver.FuncNameLists.REDUCE, lambda: client.ServerProxy("http://127.0.0.1:9120/", allow_none=True).adaptive_close_no_block() if gestureObs._cali_flag == True else None, duration=1.5, volatuationData=0.06)
    # gestureObs.register_callback(GestureObserver.FuncNameLists.OPEN, lambda: client.ServerProxy("http://127.0.0.1:9120/", allow_none=True).open_no_block() if gestureObs._cali_flag == True else None, duration=1.5)
    # gestureObs.register_callback(GestureObserver.FuncNameLists.CLOSE, lambda: client.ServerProxy("http://127.0.0.1:9120/", allow_none=True).adaptive_close_no_block() if gestureObs._cali_flag == True else None, duration=1.5)

    # 比yeah开始,牛
    # gestureObs.register_callback(GestureObserver.FuncNameLists.VICTORY, bodyObs.start_record_and_cali, duration=2.5)
    gestureObs.register_callback(GestureObserver.FuncNameLists.VICTORY, gestureObs.start_record_and_cali, duration=2.5)

    # gestureObs.register_callback(GestureObserver.FuncNameLists.THUMB_UP, lambda : client.ServerProxy("http://192.168.40.216:9120/", allow_none=True).setData() if bodyObs._cali_flag else None, duration=1)
    gestureObs.register_callback(GestureObserver.FuncNameLists.THUMB_UP, gestureObs.stop_record_and_cali, duration=1)
    # gestureObs.register_callback(GestureObserver.FuncNameLists.THUMB_UP, lambda: client.ServerProxy("http://127.0.0.1:9121/", allow_none=True).setData() if gestureObs._cali_flag == True else None, duration=1)
    # gestureObs.register_callback(GestureObserver.FuncNameLists.THUMB_UP, bodyObs.stop_record_and_cali, duration=1)



    # bodyObs = BodyObserver(imageDeque, realSense)

    fAHandle = FingerAnglesHandle()



    gestureMarker.add_observer(gestureObs)
    # poseMarker.add_observer(bodyObs)
    # *以线程方式运行观察者
    gestureObsThread = Thread(target=gestureObs.updata, args=(gestureMarker,), daemon=True, name="gestureObsThread")
    # gestureObsThread.start()

    # bodyObsThread = Thread(target=bodyObs.updata, args=(poseMarker,), daemon=True, name="bodyObsThread")
    # bodyObsThread.start()

    # showHandAngleThread = Process(target=showHandAngle, args=(gestureMarker, fAHandle,), daemon=True, name="showHandAngleThread")
    # showHandAngleThread.start()

    fps = FPS()
    timestamp = int(time.time() * 1000)
    try:

        while True:
            fps.refresh()
            
            # *获取数据帧
            # depth_image, color_image = realSense.get_frame()
            # *获取对齐数据帧
            # depth_image, color_image = realSense.get_align_frame()
            depth_image, color_image = gemini2.get_align_frame()


            # handMarker.detect_async(color_image, timestamp)
            # poseMarker.detect_async(color_image, depth_image, time.time_ns() // 1_000_000)
            gestureMarker.detect_async(color_image, depth_image, time.time_ns() // 1_000_000)

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape

            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                images = np.hstack((resized_color_image, depth_colormap))
            else:
                images = np.hstack((color_image, depth_colormap))

            # Show images
            FPS.putFPSToImage(images, fps.fps)

            # if handMarker.output_image is not None:
            #     cv2.imshow("hand", handMarker.output_image)
            if poseMarker.output_image is not None:
                # cv2.imshow("pose", poseMarker.output_image)
                # if bodyObs.get_center_frame() is not None:
                     
                    pass
            if gestureMarker.output_image is not None:
                fAHandle.updata(handWorldLandmarks2List(gestureMarker.result.hand_world_landmarks.toList()))
                # drawFingerAngleOnImage(gestureMarker.output_image, fAHandle.drawFingerAngleDatas)
                fAHandle.drawAllFingerAngleOnImage(gestureMarker.output_image, gestureMarker.result.hand_landmarks.getAllJointPoint())
                cv2.putText(gestureMarker.output_image, f"Dis:{round(gestureMarker.get_thumb_indexfinger_tip_dis(),3)}m", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
                cv2.putText(gestureMarker.output_image, f"Gesture:{gestureMarker.result.gestures.category_name}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
                cv2.imshow("gesture", gestureMarker.output_image)

            # if poseMarker.output_image is not None and gestureMarker.output_image is not None:
            #     downImage = np.hstack((poseMarker.output_image, gestureMarker.output_image))
            #     images = np.vstack((images, downImage))
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
            timestamp += 1

    finally:
        poseMarker.landmarker.close()
        # Stop streaming
        # realSense.pipeline.stop()
        gemini2.pipeline.stop()
        print("End")