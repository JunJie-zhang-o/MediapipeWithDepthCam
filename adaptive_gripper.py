#!/usr/bin/env python3
# coding=utf-8
'''
Author       : Jay jay.zhangjunjie@outlook.com
Date         : 2024-08-22 22:42:35
LastEditTime : 2024-09-18 19:01:46
LastEditors  : jay.jetson jay.zhangjunjie@outlook.com
Description  : Adapter Gripper. support multiple sensors. #!dont test
'''


import logging
import time
from threading import Thread
from xmlrpc.server import SimpleXMLRPCServer

import cv2
# import fire

from libs.ad_grip.adaptive_gripper import AdaptiveGripper, xmlRPC_start_info
from libs.sdk.findcams import getCameraIndex
from libs.sdk.tactile.cap_reader import CapReader
from libs.sdk.tactile.defines import Formats
from Robotiq.HandE import HandEForRtu

# todo:参数定值时,可以测试出来对应的型变量
# todo:如何支持双传感器
# todo:多相机同步




ESC = 27



if __name__ == "__main__":

    logging.basicConfig(level=logging.NOTSET)
    
    xmlRpcServerHost = "0.0.0.0"
    xmlRpcServerPort = 9120
    server = SimpleXMLRPCServer((xmlRpcServerHost, xmlRpcServerPort), allow_none=True)
    server.register_introspection_functions()

    # you need to create AdaptiveGripper object
    hande = HandEForRtu(port="/dev/ttyUSB0")
    # hande.mov = lambda pos, block: hande.move(pos=pos, speed=20,force=20, block=block)
    hande.mov = lambda pos, block: hande.move(pos=pos, speed=0, force=0, block=block)
    
    adGripper = AdaptiveGripper(gripper=hande, minPos=45, maxPos=0, maxDiff=25, maxDiffNum=100)
    # adGripper = AdaptiveGripper(gripper=hande, minPos=45, maxPos=0, maxDiff=25, maxDiffNum=100)
    # adGripper = AdaptiveGripper(gripper=hande, minPos=45, maxPos=0, maxDiff=35, maxDiffNum=100)
    # adGripper = AdaptiveGripper(gripper=hande, minPos=45, maxPos=0, maxDiff=30, maxDiffNum=100)
    # adGripper = AdaptiveGripper(gripper=hande, minPos=45, maxPos=0, maxDiff=40, maxDiffNum=100)
    # adGripper = AdaptiveGripper(gripper=hande, minPos=45, maxPos=0, maxDiff=70, maxDiffNum=100)
    # adGripper = AdaptiveGripper(gripper=hande, minPos=45, maxPos=0, maxDiff=50, maxDiffNum=100)
    # adGripper = AdaptiveGripper(gripper=hande, minPos=45, maxPos=0, maxDiff=55, maxDiffNum=100)
    # adGripper.SHOW_IMAGE = True
    adGripper.SHOW_IMAGE = False


    index = getCameraIndex()

    cap1 = CapReader(index[0])
    cap1.setFormats(Formats.MJPG_640_480_30)
    cap1.printFormats()
    # paddings = [29*1.7, 29, 29*1.5, 29*1.5]
    # cap1._setWarpParams([[215, 147], [415, 147], [400, 326], [230,325]], padding=paddings)
    paddings = [29*1.7, 29*1.2, 29*1.5, 29*1.5]
    cap1._setWarpParams([[224, 150], [415, 152], [385, 323], [244,320]], padding=paddings)
    print(cap1._warpParams)
    cap1.setVaildFrameSize(240, 240)
    cap1.start()

    cap2 = CapReader(index[1])
    cap2.setFormats(Formats.MJPG_640_480_30)
    cap2.printFormats()
    # paddings = [29*1.7, 29, 29*1.5, 29*1.5]
    # cap2._setWarpParams([[215, 147], [415, 147], [400, 326], [230,325]], padding=paddings)
    paddings = [29*1.7, 29*1.2, 29*1.5, 29*1.5]
    cap2._setWarpParams([[227, 152], [410, 148], [392, 315], [252,318]], padding=paddings)
    print(cap2._warpParams)
    cap2.setVaildFrameSize(240, 240)
    cap2.start()
    #  lambda

    def showImage():
        while 1:
            print(cap1.fps, cap2.fps)
            image1 = cap1.getOneFrame()
            image2 = cap2.getOneFrame()
            cv2.imshow("image1", image1)
            cv2.imshow("image2", image2)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
        # cap1.release()
        # cap2.release()
    # showImage()
    

    adGripper.registerCapGetFrame(cap1.getOneFrame)
    adGripper.registerCapGetFrame(cap2.getOneFrame)

    # adGripper.registerCapGetFrame(lambda : warp_perspective(cap1.getOneVaildFrame()))
    # adGripper.registerCapGetFrame(lambda : warp_perspective(cap2.getOneVaildFrame()))
    server.register_instance(adGripper)
    xmlRPC_start_info(xmlRpcServerHost, xmlRpcServerPort)
    # server.serve_forever()
    Thread(target=server.serve_forever, args=(), name="XMLRPC Server Thread: AdaptiveGripper", daemon=True).start()
    # Thread(target=showImage, args=(), name="XMLRPC Server Thread: showimage", daemon=True).start()
    # pass
    while 1:
        # image1 = cap1.getOneFrame()
        # image2 = cap2.getOneFrame()
        # if cap1._frame is not None  and cap2._frame is not None:
        #     cv2.imshow("image1", cap1._frame)
        #     cv2.imshow("image2", cap2._frame)
        # print(cap1.fps, cap2.fps)
        if adGripper._diffFrame2 is not None:
            cv2.imshow("showrecF1", adGripper._showrecoredFrame1)
            cv2.imshow("showrecF2", adGripper._showrecoredFrame2)
            cv2.imshow("showF1", adGripper._showFrame1)
            cv2.imshow("showF2", adGripper._showFrame2)
            cv2.imshow("diffF1", adGripper._diffFrame1)
            cv2.imshow("diffF2", adGripper._diffFrame2)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break
        else:
            time.sleep(0.001)
    
        # print("--")