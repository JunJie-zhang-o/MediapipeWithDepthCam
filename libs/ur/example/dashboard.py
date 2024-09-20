#!/usr/bin/env python3
# coding=utf-8
"""
Author       : Jay jay.zhangjunjie@outlook.com
Date         : 2024-07-04 09:35:44
LastEditTime : 2024-08-07 11:26:12
LastEditors  : Jay jay.zhangjunjie@outlook.com
Description  : Dashboard Interface
"""
import time

from ur.eseries import URERobot
from ur.eseries.script import *

if __name__ == "__main__":

    ur = URERobot(ip="192.168.1.103", autoConnect=True)
    dash = ur.createDashboardInterface()
    # dash.downloadProgram("servoj.urp","/programs/translation_sample_servoj.urp")

    # 在上传或者下载前,请确保可以通过ssh可以进行链接并已经正常有对应idkey
    dash.uploadProgram(
        "examples/servoj/urp/translation_sample_servoj.txt",
        remotePath="/programs/20240722_servoj_base.txt",
    )
    dash.uploadProgram(
        "examples/servoj/urp/translation_sample_servoj.script",
        remotePath="/programs/20240722_servoj_base.script",
    )
    dash.uploadProgram(
        "examples/servoj/urp/translation_sample_servoj.urp",
        remotePath="/programs/20240722_servoj_base.urp",
    )

    dash.popup("popup")
    dash.running()
    dash.load("testdash.urp")
    dash.load()
    dash.play()
    dash.stop()
    dash.pause()
    dash.shutdown()
    dash.running()
    dash.quit()
    print("-")
    while 1:
        time.sleep(5)
