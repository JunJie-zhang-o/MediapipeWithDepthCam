#!/usr/bin/env python3
# coding=utf-8
'''
Author       : Jay jay.zhangjunjie@outlook.com
Date         : 2024-08-20 21:23:27
LastEditTime : 2024-08-23 00:18:14
LastEditors  : Jay jay.zhangjunjie@outlook.com
Description  : Adapter Gripper. 
'''






import signal
import time
from threading import Event, Thread

import cv2
import numpy as np
from cv2 import VideoCapture



def warp_perspective(image):
    # src_points    = [[182, 119], [461, 122], [437, 368], [207, 366]]
    src_points    = [[205, 132], [429, 134], [416, 320], [219, 321]]
    output_sz = [240, 240] 
    src = np.array(src_points, dtype=np.float32)
    dst = np.array([[0, 0], [output_sz[1]-1, 0], [output_sz[1]-1, output_sz[0]-1], [0, output_sz[0]-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (output_sz[1], output_sz[0]))
    return warped


def getOneFrame(cap):
    while 1:
        ret, frame = cap.read()
        if ret:
            warpedFrame = warp_perspective(frame, )
            return warpedFrame



class AdaptiveGripper:

    SHOW_IMAGE = False

    def __init__(self, cap:VideoCapture, gripper: object, minPos, maxPos, maxDiff: int, maxDiffNum: int) -> None:
        self._cap = cap
        self._gripper = gripper

        self._minPos, self._maxPos = minPos, maxPos
        self._maxDiff, self._maxDiffNum = maxDiff, maxDiffNum

        self._openEvent, self._closeEvent = Event(), Event()
        self._openThread, self._closeThread = None, None

        Thread(target=self._capReadThread, args=(), name="Adaptive Gripper Cap Read Thread", daemon=True).start()


    def _capReadThread(self):
        while True:
            t = time.time()
            ret, frame = self._cap.read()
            if not ret:
                print("error")
                continue
            self.frame = frame
            # print(time.time() - t)
            # 0.03左右,可以达到30帧，cpu占用单核20左右


    def getOneFrame(self, arg):
        return warp_perspective(self.frame)



    def adaptiveOpen(self, *, block=True, maxPos: float=None, maxDiff: int=None, maxDiffNum: int=None):

        self._gripper.stop()
        pos = self._maxPos if maxPos is None else maxPos
        diff = self._maxDiff if maxDiff is None else maxDiff
        diffNum = self._maxDiffNum if maxDiffNum is None else maxDiffNum
        if self._closeThread is not None and self._closeThread.is_alive():
            self._closeEvent.set()
            self._closeThread.join()
            self._closeEvent.clear()

        recordFrame = self.getOneFrame(self._cap)

        self._gripper.mov(pos, block=False)
        print("Open Start")
        def adaptiveMove():
            while True:
                frame = self.getOneFrame(self._cap)
                absdiff = cv2.absdiff(frame, recordFrame)
                absDiffNum = np.sum(absdiff >= diff)
                print(f"AdaptiveOpen:{diff},{absDiffNum}")
                if absDiffNum >= diffNum:
                    self._gripper.stop()
                    if self.SHOW_IMAGE:
                         cv2.destroyWindow("Adaptive Gripper Open")
                    print("AdaptiveOpen detected")
                    break

                if self.SHOW_IMAGE:
                    # todo: 线程中showImage后,destoryWindow,可能会导致下一周期卡在showImage中
                    showImage = np.hstack((recordFrame, frame))
                    cv2.imshow("Adaptive Gripper Open", showImage)
                    key = cv2.waitKey(1) & 0xFF

                    if key == 27:
                        cv2.destroyWindow("Adaptive Gripper Open")
                        print("AdaptiveOpen keyEvent")
                        break
                if self._openEvent.is_set():
                    if self.SHOW_IMAGE:
                         cv2.destroyWindow("Adaptive Gripper Open")
                    print("AdaptiveOpen _closeEvent")
                    break

        if block:
            adaptiveMove()
        else:
            self._openThread = Thread(target=adaptiveMove, args=(),name="Adaptive Gripper Open", daemon=True)
            self._openThread.start()


    def adaptiveClose(self, block=True, minPos: float=None, maxDiff: int=None, maxDiffNum: int=None):
        t1 = time.time()
        self._gripper.stop()
        pos = self._minPos if minPos is None else minPos
        diff = self._maxDiff if maxDiff is None else maxDiff
        diffNum = self._maxDiffNum if maxDiffNum is None else maxDiffNum
        t1 = time.time()
        if self._openThread is not None and self._openThread.is_alive():
            self._openEvent.set()
            self._openThread.join()
            self._openEvent.clear()

        recordFrame = self.getOneFrame(self._cap)
        cv2.imwrite("recordFrame.jpeg", recordFrame)
        t4 = time.time()

        
        self._gripper.mov(pos, block=False)
        print("Close Start")
        def adaptiveMove():
            while True:

                frame = self.getOneFrame(self._cap)
                cv2.imwrite("frame.jpeg", frame)
                absdiff = cv2.absdiff(frame, recordFrame)
                absDiffNum = np.sum(absdiff >= diff)
                print(f"AdaptiveClose:{diff},{absDiffNum}")
                if absDiffNum >= diffNum:
                    self._gripper.stop()
                    if self.SHOW_IMAGE:
                        cv2.destroyWindow("Adaptive Gripper Close")
                    print("AdaptiveClose detected")
                    break

                if self.SHOW_IMAGE:
                    # todo: 线程中showImage后,destoryWindow,可能会导致下一周期卡在showImage中
                    showImage = np.hstack((recordFrame, frame))
                    cv2.imshow("Adaptive Gripper Close", showImage)
                    key = cv2.waitKey(1) & 0xFF

                    if key == 27:
                        cv2.destroyWindow("Adaptive Gripper Close")
                        print("AdaptiveClose keyEvent")
                        break
                if self._closeEvent.is_set():
                    if self.SHOW_IMAGE:
                        cv2.destroyWindow("Adaptive Gripper Close")
                    print("AdaptiveClose _closeEvent")
                    break

            
        if block:
            adaptiveMove()
        else:
            self._closeThread = Thread(target=adaptiveMove, args=(),name="Adaptive Gripper Close", daemon=True)
            self._closeThread.start()


    def open(self, block=True):
        if self._closeThread is not None and self._closeThread.is_alive():
            self._closeEvent.set()
            self._closeThread.join()
            self._closeEvent.clear()
        self._gripper.mov(pos=self._maxPos, block=block)


    def close(self, block=True):
        if self._openThread is not None and self._openThread.is_alive():
            self._openEvent.set()
            self._openThread.join()
            self._openEvent.clear()
        self._gripper.mov(pos=self._minPos, block=block)

