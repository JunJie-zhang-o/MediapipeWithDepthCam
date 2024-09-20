#!/usr/bin/env python3
# coding=utf-8
'''
Author       : error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date         : 2024-08-22 14:57:43
LastEditTime : 2024-08-22 15:51:51
LastEditors  : error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Description  : 
'''



from xmlrpc.server import SimpleXMLRPCServer
import numpy as np
from Robotiq.HandE import HandEForRtu
from adaptive_grip import AdaptiveGripper
import cv2


cap = cv2.VideoCapture(1)

class Cap:


    def __init__(self, index):
        self._cap = cv2.VideoCapture(index)


    def getOneFrame(self):
        while 1:
            ret, frame = self._cap.read()
            if ret:
                cv2.imshow("frame1", frame)

                warpedFrame = warp_perspective(frame, )
                cv2.imshow("warped", warpedFrame)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    # break
                    cv2.destroyAllWindows()
                    return warpedFrame
    
        cv2.destroyAllWindows()


    





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
            cv2.imshow("frame1", frame)

            warpedFrame = warp_perspective(frame, )
            cv2.imshow("warped", warpedFrame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                # break
                cv2.destroyAllWindows()
                return warpedFrame
    
    cv2.destroyAllWindows()
    


# cap.getOneFrame = lambda: getOneFrame(cap)

gripper = HandEForRtu(port="/dev/ttyUSB1")
gripper.move(0,50,0,True)
gripper.mov = lambda pos, block: gripper.move(pos=pos, speed=0, force=0, block=block)

adGripper = AdaptiveGripper(cap=cap, gripper=gripper, maxPos=0, minPos=45, maxDiff=40, maxDiffNum=250)

server = SimpleXMLRPCServer(("0.0.0.0", 9120), allow_none=True)
server.register_instance(adGripper)
print(f"Listening on port 9120...")
server.serve_forever()
