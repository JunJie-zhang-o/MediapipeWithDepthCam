#!/usr/bin/env python3
# coding=utf-8
'''
Author       : Jay jay.zhangjunjie@outlook.com
Date         : 2024-08-07 16:07:41
LastEditTime : 2024-09-20 16:03:52
LastEditors  : jay jay.zhangjunjie@outlook.com
Description  : 数据平滑器
'''

from collections import deque


class MovingAverage3D:
    """
        移动窗口平均法,用于平滑发送给机器人的x,y,z数据
    """
    def __init__(self, windowsSize:int=5) -> None:
        
        self._windowsSize = windowsSize
        self._x = deque(maxlen=windowsSize)
        self._y = deque(maxlen=windowsSize)
        self._z = deque(maxlen=windowsSize)


    def handle(self, x, y, z):
        return self.__handle(x, y, z)
    

    # def handle(self, data) -> tuple[float, float, float] | None:
    #     return self.__handle(data[0], data[1], data[2])
    

    def __handle(self, x, y, z) :
        self._x.append(x)
        self._y.append(y)
        self._z.append(z)

        if self._x.__len__() == self._windowsSize:
            return sum(self._x)/self._windowsSize, sum(self._y)/self._windowsSize, sum(self._z)/self._windowsSize





if __name__ == "__main__":

    pass


