#!/usr/bin/env python3
# coding=utf-8
"""
Author       : Jay jay.zhangjunjie@outlook.com
Date         : 2024-08-03 23:19:02
LastEditTime : 2024-08-04 00:50:38
LastEditors  : Jay jay.zhangjunjie@outlook.com
Description  : 
"""


from dataclasses import dataclass
import time
import numpy as np
import cv2


@dataclass
class FingerData:
    first: float = None
    second: float = None
    third: float = None


    def __getitem__(self, key:int):
        if type(key) is int:
            ret = (self.first, self.second, self.third)
            return ret[key]
        else:
            raise Exception("The DrawFingerAngleData Class Getitem Error: The Key Mest be Int!!!")


@dataclass
class HandAngleData:
    thumb: FingerData
    indexFinger: FingerData
    middleFinger: FingerData
    ringFinger: FingerData
    pinkyFinger: FingerData

    def __getitem__(self, key:int):
        if type(key) is int:
            ret = (self.thumb, self.indexFinger, self.middleFinger, self.ringFinger, self.pinkyFinger)
            return ret[key]
        else:
            raise Exception("The DrawFingerAngleData Class Getitem Error: The Key Mest be Int!!!")


@dataclass
class DrawFingerAngleData:
    x:float
    y:float
    angle:float


    def __getitem__(self, key:int):
        if type(key) is int:
            ret = (self.x, self.y, self.angle)
            return ret[key]
        else:
            raise Exception("The DrawFingerAngleData Class Getitem Error: The Key Mest be Int!!!")

# class DrawHandAngleData:




def handWorldLandmarks2List(data):

    def getXYZ(d):
        return np.asanyarray([d['x'], d['y'], d['z']])

    ret = []
    for i in range(5):
        ret.append(
            [
                getXYZ(data[0]),
                getXYZ(data[i * 4 + 1]),
                getXYZ(data[i * 4 + 2]),
                getXYZ(data[i * 4 + 3]),
                getXYZ(data[i * 4 + 4]),
            ]
        )
    return ret


class FingerAnglesHandle:

    def __init__(self) -> None:

        self._recordData = None
        self._handAngleData = None
        self.drawFingerAngleDatas = None

    def updata(self, data: list):
        self._recordData = data
        fingersAngle = []
        drawFingerAngleDatas = []
        for _type in data:
            # fingerAngles = [float(_type[1][0]), float(_type[1][1]), float(_type[1][2])]
            fingerAngles = []
            for index in range(3):
                v1 = self.__getVector(_type[index], _type[index + 1])
                v2 = self.__getVector(_type[index + 1], _type[index + 2])
                angle = self.__getAngleFromVectors(v1, v2)
                fingerAngles.append(angle)
                drawFingerAngleDatas.append(DrawFingerAngleData(_type[index + 1][0],_type[index + 1][1],angle))
            fingersAngle.append(FingerData(*fingerAngles))
        self._handAngleData = HandAngleData(*fingersAngle)
        self.drawFingerAngleDatas = drawFingerAngleDatas.copy()
        return self._handAngleData
    

    def drawFingerAngleOnImage(self, rgbImage, drawData, angle):
        t = time.time()
        cv2.putText(rgbImage, f"{angle}", (int(drawData[0]*rgbImage.shape[1]), int(drawData[1]*rgbImage.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        print(time.time() - t)

    def drawAllFingerAngleOnImage(self, rgbImage, drawData):
        """绘制所有的手指关节角度

        Args:
            rgbImage (_type_): 要绘制的图片
            drawData (_type_): 绘制的手指关节点,[[[手指1关节1的x,手指1关节1的y],[手指1关节2的x,手指1关节2的y]],[[手指1关节1的x,手指1关节1的y],[手指1关节2的x,手指1关节2的y]]]
        """
        fingerIndex, jointIndex = 0,0
        for fingerData in drawData:
            for oneLocData in fingerData:
                # cv2.putText(rgbImage, f"{self._handAngleData[jointIndex][0]}\n{self._handAngleData[jointIndex][1]}\n{self._handAngleData[jointIndex][2]}", (int(fingerData[0][0]*rgbImage.shape[1]), int(fingerData[0][1]*rgbImage.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(rgbImage, f"   {int(self._handAngleData[jointIndex][fingerIndex])}", (int(oneLocData[0]*rgbImage.shape[1]), int(oneLocData[1]*rgbImage.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                fingerIndex = fingerIndex+1
            jointIndex = jointIndex+1
            fingerIndex = 0


    def __getVector(self, startP, endP):

        dirVector = endP - startP
        norm = np.linalg.norm(dirVector)
        if norm == 0:
            return dirVector
        normVector = dirVector / norm
        return normVector

    def __getAngleFromVectors(self, vectorA, vectorB):
        dotProduct = np.dot(vectorA, vectorB)
        normA = np.linalg.norm(vectorA)
        normB = np.linalg.norm(vectorB)
        # 计算夹角的余弦值
        cosTheta = dotProduct / (normA * normB)
        # 计算夹角（弧度）
        angleRad = np.arccos(cosTheta)
        # 将夹角转换为角度
        angleDeg = np.degrees(angleRad)
        return round(float(angleDeg), 4)






def drawFingerAngleOnImage(rgbImage, drawDatas:list):
    """
        绘制手指的角度

        The drawDatas:[ [x,y,angle], [x,y,angle], [x,y,angle] ] 
    """
    for i in drawDatas:
        cv2.putText(rgbImage, f"Angle:{i[2]}", (i[0], i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)