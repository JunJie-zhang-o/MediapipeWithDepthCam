#!/usr/bin/env python3
# coding=utf-8
'''
Author       : Jay jay.zhangjunjie@outlook.com
Date         : 2024-07-14 14:47:25
LastEditTime : 2024-08-07 11:47:42
LastEditors  : Jay jay.zhangjunjie@outlook.com
Description  : Primary Interface
'''
import time

from ur.eseries import URERobot

if __name__ == "__main__":


    ip = "192.168.10.109"
    robot = URERobot(ip, True)
    priMonitor = robot.createPrimaryMonitorInterface()
    priMonitor.monitorStart()

    while 1:
        print(priMonitor.ProgramThreadsMessage)
        time.sleep(2)