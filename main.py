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


    # realSense = RealSense(framerate=60)
    # realSense.set_align_mode()

    gemini2 = Gemini2()
    gemini2.set_align_mode()

    # handMarker = HandLandMark(model_path="model/hand_landmarker.task")
    poseMarker = PoseLandMarkDetector(model_path="model/pose_landmarker_full.task",
                                      min_pose_detection_confidence=0.9,
                                      min_pose_presence_confidence=0.9,
                                      min_tracking_confidence=0.9)
    # poseMarker = PoseLandMarkDetector(model_path="model/pose_landmarker_heavy.task")
    gestureMarker = GestureLandMarkDetector(model_path="model/gesture_recognizer.task")

    gestureObs = GestureObserver()
    imageDeque = ImageList()
    bodyObs = BodyObserver(imageDeque, gemini2)

    gestureObs.register_callback(GestureObserver.FuncNameLists.INCREASE, lambda:print("inc"), duration=2, volatuationData=0.04)
    gestureObs.register_callback(GestureObserver.FuncNameLists.REDUCE, lambda:print("dec"), duration=2, volatuationData=0.04)

    gestureObs.register_callback(GestureObserver.FuncNameLists.VICTORY, bodyObs.start_record_and_cali, duration=2, volatuationData=0)
    gestureObs.register_callback(GestureObserver.FuncNameLists.THUMB_UP, bodyObs.stop_record_and_cali, duration=2, volatuationData=0)



    # bodyObs = BodyObserver(imageDeque, realSense)

    fAHandle = FingerAnglesHandle()



    # gestureMarker.add_observer(gestureOsb)
    poseMarker.add_observer(bodyObs)
    # *以线程方式运行观察者
    gestureObsThread = Thread(target=gestureObs.updata, args=(gestureMarker,), daemon=True, name="gestureObsThread")
    gestureObsThread.start()

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
            # z = realSense.get_depth_value(315, 191, depth_image)
            # point =  realSense.get_actual_pose(315, 191, z)
            # print(point)

            # print(time.time()* 1000)
            # handMarker.detect_async(color_image, timestamp)
            # imageDeque.set_image_data(color_image, depth_image, timestamp)
            poseMarker.detect_async(color_image, depth_image, timestamp)
            gestureMarker.detect_async(color_image, depth_image, timestamp)

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
                cv2.imshow("pose", poseMarker.output_image)
                pass
            if gestureMarker.output_image is not None:
                fAHandle.updata(handWorldLandmarks2List(gestureMarker.result.hand_world_landmarks.toList()))
                # # drawFingerAngleOnImage(gestureMarker.output_image, fAHandle.drawFingerAngleDatas)
                fAHandle.drawAllFingerAngleOnImage(gestureMarker.output_image, gestureMarker.result.hand_landmarks.getAllJointPoint())
                # cv2.putText(gestureMarker.output_image, f"Dis:{round(gestureMarker.get_thumb_indexfinger_tip_dis(),3)}m", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
                cv2.imshow("gesture", gestureMarker.output_image)
                pass

                # images = np.vstack((images,np.hstack((poseMarker.output_image, gestureMarker.output_image))))
                # images = cv2.resize(images, None, fx=0.5, fy=0.5)
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