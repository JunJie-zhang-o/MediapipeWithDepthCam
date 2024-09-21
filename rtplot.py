#!/home/jetson/miniconda3/envs/py311/bin/python
# coding=utf-8
'''
Author       : jay.jetson jay.zhangjunjie@outlook.com
Date         : 2024-09-04 03:23:09
LastEditTime : 2024-09-20 16:41:54
LastEditors  : jay jay.zhangjunjie@outlook.com
Description  : 
'''
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from threading import Event
from typing import Callable
import warnings

import numpy as np
from libs.utils.plot.rtplot import PlotWindowType, RTMessage, RealTimePlot, Suber
import pyqtgraph as pg
import zmq
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow
from pyqtgraph.Qt import QtGui

# 禁用科学记数法
np.set_printoptions(suppress=True)


if __name__ == '__main__':


    suber = Suber("tcp://127.0.0.1:5556")
    # suber = Suber("tcp://127.0.0.1:5557")

    app = QApplication(sys.argv)



    rtPlot = RealTimePlot("Data", 10, suber)
    rtPlot.setUpdateTrigger(suber)
    

    def xFunc(rtMsg:RTMessage):
        x = float(rtMsg.message.split(",")[0])
        # if abs(x) <= 1.2:
        #     x = 0
        return rtMsg.totalTime, x

    def yFunc(rtMsg: RTMessage):
        return rtMsg.totalTime, float(rtMsg.message.split(",")[1])

    def zFunc(rtMsg: RTMessage):
        return rtMsg.totalTime, float(rtMsg.message.split(",")[2])


    rtPlot.addSubWindow(title="X2", callback=xFunc, row=1,col=1, yRange=(-10,10), windowType=PlotWindowType.ROLL_WINDOW)
    rtPlot.addSubWindow(title="Y2", callback=yFunc, row=2,col=1, yRange=(-10,10), windowType=PlotWindowType.ROLL_WINDOW)
    rtPlot.addSubWindow(title="Z2", callback=zFunc, row=3,col=1, yRange=(-10,10), windowType=PlotWindowType.ROLL_WINDOW)


    # 退出时自动保存数据
    def on_exit():
        rtPlot.save_all_data()
        suber.stop()
    # app.aboutToQuit.connect(lambda: rtPlot.save_all_data())
    suber.start()
    rtPlot.show()
    sys.exit(app.exec_())
    suber.stop()
