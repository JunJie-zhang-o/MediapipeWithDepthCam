#!/usr/bin/env python3
# coding=utf-8
"""
Author       : Jay jay.zhangjunjie@outlook.com
Date         : 2024-08-07 15:52:42
LastEditTime : 2024-08-07 19:00:17
LastEditors  : Jay jay.zhangjunjie@outlook.com
Description  : 数据过滤器
"""


from abc import ABC, abstractmethod


class Limiter(ABC):

    @abstractmethod
    def limit(self, data: float):
        pass


class HighLimiter(Limiter):
    """
    高于阈值被限制为阈值
    """

    def __init__(self, threshold) -> None:
        self._threshold = threshold

    def limit(self, data: float) -> float:
        return min(data, self._threshold)


class LowLimiter(Limiter):
    """
    低于阈值被限制为阈值
    """

    def __init__(self, threshold: float) -> None:
        self._threshold = threshold

    def limit(self, data: float) -> float:
        return max(data, self._threshold)


class BandLimiter(Limiter):
    """
    小于高阈值, 大于低阈值
    """

    def __init__(self, highThreshold: float, lowThreshold: float) -> None:
        self._highThreshold = highThreshold
        self._lowThreshold = lowThreshold

    def limit(self, data):
        temp = min(data, self._highThreshold)
        temp = max(temp, self._lowThreshold)
        return temp


class DValueLimiter(Limiter):
    """
    差值限制,主要用于处理数据跳动过大的情况
    """

    def __init__(self, maxDiffThreshold: float) -> None:
        self._maxDiffThreshold = maxDiffThreshold
        self._lastData = None
    
    def limit(self, data: float):
        if self._lastData is None:
            self._lastData = data

        diff = abs(data - self._lastData)
        ret = self._lastData  if diff >= self._maxDiffThreshold else data
        self._lastData = ret
        return ret
