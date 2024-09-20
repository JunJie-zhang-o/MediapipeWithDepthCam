#!/usr/bin/env python3
# coding=utf-8
'''
Author       : Jay jay.zhangjunjie@outlook.com
Date         : 2024-08-22 22:42:35
LastEditTime : 2024-09-04 16:54:47
LastEditors  : Jay jay.zhangjunjie@outlook.com
Description  : Adapter Gripper. support multiple sensors. #! tested
'''


import logging
import time
import warnings
from threading import Event, Thread
from typing import Callable


import cv2
import fire
import numpy as np

# todo:参数定值时,可以测试出来对应的型变量
# todo:如何支持双传感器
# todo:多相机同步




logger = logging.getLogger("ViTai:AdaptiveGripper")

ESC = 27


class AdaptiveGripper:
    """
        How to Use:

        >>> hande = HandEForRtu(port="/dev/ttyUSB1")

        >>> hande.mov = lambda pos, block: hande.move(pos=pos, speed=0, force=0, block=block)

        >>> # the gripper obj must have the stop and mov func. and the stop func dont have the params, the mov func can pass in two parameters are pos and block。

        >>> adGripper = AdaptiveGripper(gripper=hande, minPos=45, maxPos=0, maxDiff=30, maxDiffNum=250)

        >>> def getOneFrame(cap):
        >>>     while 1:
        >>>         ret, frame = cap.read()
        >>>         if ret:
        >>>             warpedFrame = warp_perspective(frame, )
        >>>             return warpedFrame

        >>> adGripper.registerCapGetFrame(getOneFrame)  # the callback need to return the vaild realtime frame from the sensor

        >>> adgripper.open()

        >>> adgripper.close()

        >>> adgripper.adaptiveOpen()

        >>> adgripper.adaptiveClose()

    """
    SHOW_IMAGE = False

    def __init__(self, gripper: object, minPos: float, maxPos: float, maxDiff: int, maxDiffNum: int) -> None:
        """
        _summary_

        Args:
            gripper (object): _description_
            minPos (float): _description_
            maxPos (float): _description_
            maxDiff (int): _description_
            maxDiffNum (int): _description_
        """
        self._gripper = gripper

        self._capGetOneFrameCallbacks = []

        self._minPos, self._maxPos = minPos, maxPos
        self._maxDiff, self._maxDiffNum = maxDiff, maxDiffNum

        self._openEvent, self._closeEvent = Event(), Event()
        self._openThread, self._closeThread = None, None

    
    def registerCapGetFrame(self, callback:Callable):
        """
        register the callback functions for get one vaild frame from vision tactail camera.

            you can register multiple callback functions when there are multiple sensors

        Args:
            callback (Callable): _description_
        """
        self._capGetOneFrameCallbacks.append(callback)
        if len(self._capGetOneFrameCallbacks) > 2:
            warnings.warn("You have registered more than two getFrame func for cap, please confirm whether the quantity is correct.")


    def adaptiveOpen(self, *, block=True, maxPos: float=None, maxDiff: int=None, maxDiffNum: int=None):
        """
        Adaptive opening

            the gripper will move in the direction of the finger opening, and auto stop after detecting the object.

        Args:
            block (bool, optional): When set to True, the function will return after the object is detected and the control claw stops. Otherwise, it will return immediately, and the detection will run in the background as a thread. Defaults to True.
            maxPos (float, optional): You can set the target open max pos, only this call takes effect. Defaults to None.
            maxDiff (int, optional): You can set the max diff value for images detected before and after, only this call takes effect. Defaults to None.
            maxDiffNum (int, optional): You can set the max diff num value for images detected before and after, only this call takes effect. Defaults to None.
        """
        self._gripper.stop()
        pos = self._maxPos if maxPos is None else maxPos
        diff = self._maxDiff if maxDiff is None else maxDiff
        diffNum = self._maxDiffNum if maxDiffNum is None else maxDiffNum
        if self._closeThread is not None and self._closeThread.is_alive():
            self._closeEvent.set()
            self._closeThread.join()
            self._closeEvent.clear()
            self._closeThread = None
            

        recordFrames = [getOneFrame() for getOneFrame in self._capGetOneFrameCallbacks]

        self._gripper.mov(pos, block=False)
        logger.debug("Open Start")
        def adaptiveMove():
            while True:
                frames = [getOneFrame() for getOneFrame in self._capGetOneFrameCallbacks]
                absDiffs = list(map(cv2.absdiff, frames, recordFrames))
                absDiffNums = [np.sum(absDiff >= diff) for absDiff in absDiffs]
                logger.debug(f"AdaptiveOpen:{diff},{absDiffNums}")
                if all(absDiffNum >= diffNum for absDiffNum in absDiffNums):
                    self._gripper.stop()
                    if self.SHOW_IMAGE:
                         cv2.destroyWindow("Adaptive Gripper Open")
                    logger.debug("AdaptiveOpen detected")
                    break

                if self.SHOW_IMAGE and self._openThread is None:    #! dont show image in the thread
                    hImages = []
                    for i in range(len(frames)):
                        hImages.append(np.hstack((recordFrames[i], frames[i])))
                    if len(hImages) == 1:
                        cv2.imshow("Adaptive Gripper Open", hImages[0])
                    elif len(hImages) >1:
                        showImage = np.vstack((image for image in hImages))
                        cv2.imshow("Adaptive Gripper Open", showImage)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ESC:
                        cv2.destroyWindow("Adaptive Gripper Open")
                        logger.debug("AdaptiveOpen keyEvent")
                        break
                if self._openEvent.is_set():
                    if self.SHOW_IMAGE:
                         cv2.destroyWindow("Adaptive Gripper Open")
                    logger.debug("AdaptiveOpen _closeEvent")
                    break

        if block:
            adaptiveMove()
        else:
            # todo: the close thread will duplicate creation, please confirm whether only one close thread is allowed to run.
            self._openThread = Thread(target=adaptiveMove, args=(),name="Adaptive Gripper Open", daemon=True)
            self._openThread.start()


    def adaptiveClose(self, block=True, minPos: float=None, maxDiff: int=None, maxDiffNum: int=None):
        """
        Adaptive closing

            the gripper will move in the direction of the finger closing, and auto stop after detecting the object.

        Args:
            block (bool, optional): When set to True, the function will return after the object is detected and the control claw stops. Otherwise, it will return immediately, and the detection will run in the background as a thread. Defaults to True.
            maxPos (float, optional): You can set the target open max pos, only this call takes effect. Defaults to None.
            maxDiff (int, optional): You can set the max diff value for images detected before and after, only this call takes effect. Defaults to None.
            maxDiffNum (int, optional): You can set the max diff num value for images detected before and after, only this call takes effect. Defaults to None.
        """
        self._gripper.stop()
        pos = self._minPos if minPos is None else minPos
        diff = self._maxDiff if maxDiff is None else maxDiff
        diffNum = self._maxDiffNum if maxDiffNum is None else maxDiffNum

        # stop the open thread
        if self._openThread is not None and self._openThread.is_alive():
            self._openEvent.set()
            self._openThread.join()
            self._openEvent.clear()
            self._openThread = None


        recordFrames = [getOneFrame() for getOneFrame in self._capGetOneFrameCallbacks]
        
        self._gripper.mov(pos, block=False)
        logger.debug("Close Start")
        def adaptiveMove():
            while True:
                frames = [getOneFrame() for getOneFrame in self._capGetOneFrameCallbacks]
                absDiffs = list(map(cv2.absdiff, frames, recordFrames))
                absDiffNums = [np.sum(absDiff >= diff) for absDiff in absDiffs]
                logger.debug(f"AdaptiveClose:{diff},{absDiffNums}")
                if all(absDiffNum >= diffNum for absDiffNum in absDiffNums):
                    self._gripper.stop()
                    if self.SHOW_IMAGE:
                        cv2.destroyWindow("Adaptive Gripper Close")
                    logger.debug("AdaptiveClose detected")
                    break

                if self.SHOW_IMAGE and self._closeThread is None:    #! dont show image in the thread
                    hImages = []
                    for i in range(len(frames)):
                        hImages.append(np.hstack((recordFrames[i], frames[i])))
                    showImage = np.vstack((image for image in hImages))

                    cv2.imshow("Adaptive Gripper Open", showImage)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ESC:
                        cv2.destroyWindow("Adaptive Gripper Close")
                        logger.debug("AdaptiveClose keyEvent")
                        break
                if self._closeEvent.is_set():
                    if self.SHOW_IMAGE:
                        cv2.destroyWindow("Adaptive Gripper Close")
                    logger.debug("AdaptiveClose _closeEvent")
                    break
            
        if block:
            adaptiveMove()
        else:
            # todo: the close thread will duplicate creation, please confirm whether only one close thread is allowed to run.
            self._closeThread = Thread(target=adaptiveMove, args=(),name="Adaptive Gripper Close", daemon=True)
            self._closeThread.start()


    def open(self, block=True, maxPos: float=None):
        """
        open the gripper 

        Args:
            block (bool, optional): When set to True, the function will return after the gripper move done. Otherwise, it will return immediately, just call the move func. Defaults to True.
            minPos (float, optional): You can set the target open min pos, only this call takes effect. Defaults to None.
        """
        pos = self._maxPos if maxPos is None else maxPos
        if self._closeThread is not None and self._closeThread.is_alive():
            self._closeEvent.set()
            self._closeThread.join()
            self._closeEvent.clear()
            self._closeThread = None
        self._gripper.mov(pos=pos, block=block)


    def close(self, block=True, minPos: float=None):
        """
        close the gripper

        Args:
            block (bool, optional): When set to True, the function will return after the gripper move done. Otherwise, it will return immediately, just call the move func. Defaults to True.
            minPos (float, optional): You can set the target open min pos, only this call takes effect. Defaults to None.
        """
        pos = self._minPos if minPos is None else minPos
        if self._openThread is not None and self._openThread.is_alive():
            self._openEvent.set()
            self._openThread.join()
            self._openEvent.clear()
            self._openThread = None
        self._gripper.mov(pos=pos, block=block)


def xmlRPC_start_info(host, port):
    """打印并记录

    Args:
        host (str): rpc host
        port (int): rpc port
    """
    MAX_STR_LENGTH = 20
    MAX_PRINT_LENGTH = 40
    t = []
    t.append("-".center(MAX_PRINT_LENGTH, "-"))
    t.append("XMLRPC server has been started".center(
        MAX_STR_LENGTH, " ").center(MAX_PRINT_LENGTH, "-"))
    t.append("%s"%host.center(MAX_STR_LENGTH,
                                    " ").center(MAX_PRINT_LENGTH, "-"))
    t.append("%s"%str(port).center(MAX_STR_LENGTH,
                                    " ").center(MAX_PRINT_LENGTH, "-"))
    t.append("-".center(MAX_PRINT_LENGTH, "-"))
    [print(i) for i in t]




if __name__ == "__main__":

    from xmlrpc.server import SimpleXMLRPCServer
    
    xmlRpcServerHost = "0.0.0.0"
    xmlRpcServerPort = 9120
    server = SimpleXMLRPCServer((xmlRpcServerHost, xmlRpcServerPort), allow_none=True)
    server.register_introspection_functions()

    # you need to create AdaptiveGripper object
    # adGripper = AdaptiveGripper()
    # server.register_instance(adGripper)
    xmlRPC_start_info(xmlRpcServerHost, xmlRpcServerPort)
    server.serve_forever()
    pass