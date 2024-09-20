#!/usr/bin/env python3
# coding=utf-8
'''
Author       : Jay jay.zhangjunjie@outlook.com
Date         : 2024-07-04 10:28:10
LastEditTime : 2024-09-20 16:08:02
LastEditors  : jay jay.zhangjunjie@outlook.com
Description  : 
'''


from .dash import Dashboard
from .datastruct import *
from .interpreter import Interpreter
from .primary import Primary, PrimaryMonitor
from .rtde import RTDE, ConfigFile, Recipe
from .script import *
from .secondary import Secondary, SecondaryMonitor


class URERobot:

    def __init__(self, ip: str, autoConnect: bool) -> None:
        self.ip = ip
        self.autoConnect = autoConnect


    def createPrimaryInterface(self) -> Primary:
        return Primary(self.ip, self.autoConnect)


    def createPrimaryMonitorInterface(self) -> PrimaryMonitor:
        return PrimaryMonitor(self.ip, self.autoConnect)


    def createSecondaryInterface(self) -> Secondary:
        return Secondary(self.ip, self.autoConnect)


    def createSecondaryMonitorInterface(self) -> SecondaryMonitor:
        return SecondaryMonitor(self.ip, self.autoConnect)


    def createDashboardInterface(self) -> Dashboard:
        return Dashboard(self.ip, self.autoConnect)


    def createRTDEInterface(self) -> RTDE:
        return RTDE(self.ip, self.autoConnect)


    def createInterpreterInterface(self) -> Interpreter:
        return Interpreter(self.ip, self.autoConnect)
