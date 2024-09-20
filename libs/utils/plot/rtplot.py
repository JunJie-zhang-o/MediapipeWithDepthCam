#!/usr/bin/python
# coding=utf-8
'''
Author       : jay.jetson jay.zhangjunjie@outlook.com
Date         : 2024-09-04 03:40:34
LastEditTime : 2024-09-20 11:21:20
LastEditors  : Jay jay.zhangjunjie@outlook.com
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
import pyqtgraph as pg
import zmq
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow
from pyqtgraph.Qt import QtGui

# 禁用科学记数法
np.set_printoptions(suppress=True)



"""
    Example Puber From Zmq:
    puber = zmq.Context().socket(zmq.PUB)
    puber.bind("tcp://127.0.0.1:5555")
    puber.set_hwm(100)

    while 1:
        time.sleep(0.015)
        s = f"{x},{y},{z}"
        puber.send_string(s)
"""

"""
    PUB&SUB

    通信延时:
        在本机通信的情况下,PUB和SUB设置HWM为100, 发送方0.015间隔,发送114个数据后接受方为1.710053 | 1.7101030349731445

    如何解决前几个数据包丢失的情况:
        先打开SUB,再打开PUB
        PUB打开后延时1s左右在发送
        HWM设置尽量搞一些,在本文测试中,100的HWM已经可以避免前几个数据包丢失的情况
"""



@dataclass
class RTMessage:
    timestamp:float     # 本地接受的时间戳
    totalTime:float     # 本次开始接受至本次消息到达的时间总长
    message:str         # 传输的数据,该数据中也可以添加时间戳


class Suber(QThread):

    SPLIT_CHAR = ","
    rtMsgSignal = pyqtSignal(RTMessage)

    def __init__(self, address, topic=""):
        super().__init__()
        self._address = address
        self._topic = topic
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.set_hwm(100)
        self._socket.connect(self._address)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, self._topic)
        self._stop_event = Event()

        self._message = None
        self._firstRecvT = None
        self._internal = 0
        self._timestamp = 0
        self._totalTime = 0

        self._rtMsg = None
        


    def run(self):
        print(f"Subscriber started, listening to {self._topic} on {self._address}")
        t1 = None
        while not self._stop_event.is_set():
            try:
                message = self._socket.recv_string(flags=zmq.NOBLOCK)
                if t1 is None:
                    t1 = time.perf_counter()
                    if self._firstRecvT is None:
                        self._firstRecvT = t1

                t2 = time.perf_counter()
                self._interval = t2 - t1
                self._timestamp = t2
                self._totalTime = self._timestamp - self._firstRecvT
                t1 = t2

                self._message = message


                self._rtMsg = RTMessage(timestamp=self._timestamp, totalTime=self._totalTime, message=self._message)
                self.rtMsgSignal.emit(self._rtMsg)
                
                print(f"Received message on topic {self._topic}: {message} | {self._totalTime} | {self._interval}")
            except zmq.Again:
                pass

    
    def stop(self):
        self._stop_event.set()
        self._socket.close()
        self._context.term()
        print("Subscriber stopped")


    
    def connect(self, slot):
        self.rtMsgSignal.connect(slot)




class PlotSubWindow(ABC):
    
    DEFAULT_X_RANGE = (0, 10)
    DEFAULT_Y_RANGE = (-1, 1)

    def __init__(self, title, callback, row, col, xRange, yRange, win):
        self.plot = win.addPlot(row, col, title=title)
        self.plot.setTitle(title, size="30pt")

        self.x_data = []
        self.y_data = []

        self.curve = self.plot.plot()
        self.curve.setPen(pg.mkPen(color="w", width=3))

        # 设置坐标轴标签的字体大小
        axis_font = QtGui.QFont()
        axis_font.setPointSize(16)  # 增加坐标轴字体大小
        axis_font.setBold(True)
        self.plot.getAxis('bottom').setStyle(tickFont=axis_font)
        self.plot.getAxis('bottom').setPen(pg.mkPen(color="w", width=2))
        self.plot.getAxis('left').setPen(pg.mkPen(color="w", width=2))
        self.plot.getAxis('left').setStyle(tickFont=axis_font)

        # 添加显示最大值和最小值的文本项
        self.max_text = pg.TextItem(anchor=(1, 1), color=(100,100,100))
        self.min_text = pg.TextItem(anchor=(1, 0), color=(100,100,100))
        self.max_text.setFont(axis_font)
        self.min_text.setFont(axis_font)
        self.plot.addItem(self.max_text)
        self.plot.addItem(self.min_text)

        # 添加最新数据的显示
        # self.lastValue = pg.TextItem(anchor=(0.7, 1), color=(100,100,100))
        # self.lastValue.setFont(axis_font)
        # self.plot.addItem(self.lastValue)

        # 添加鼠标数据
        # self._mouseValue = pg.TextItem(anchor=(0, 1), color=(100,100,100))
        # self._mouseValue.setFont(axis_font)
        # self.plot.addItem(self._mouseValue)

        # 鼠标横纵轴
        # self._vLine = pg.InfiniteLine(angle=90, movable=False)
        # self._hLine = pg.InfiniteLine(angle=0, movable=False)
        # self.plot.addItem(self._vLine, ignoreBounds=True)
        # self.plot.addItem(self._hLine, ignoreBounds=True)

        # self.proxy = pg.SignalProxy(self.plot.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)


        # 设置y轴的初始范围
        self._yRange = yRange
        if yRange is not None:
            self.plot.setYRange(*yRange, padding=0.1)
        else:
            self.plot.setYRange(*self.DEFAULT_Y_RANGE, padding=0.1)
        self._xRange = xRange
        if xRange is not None:
            self.plot.setXRange(*xRange, padding=0.1)
        else:
            self.plot.setXRange(*self.DEFAULT_X_RANGE, padding=0.1)


        self.callback = callback

    
    @abstractmethod
    def update(self, rtMsg):
        pass


    def save_data(self, filename):
        """保存当前窗口的数据为CSV文件"""
        data = np.column_stack((self.x_data, self.y_data))
        np.savetxt(filename, data, delimiter=",", header="x,y", comments="")
        print(f"Data saved to {filename}")


class CompressWindow(PlotSubWindow):


    def update(self, rtMsg):

        x, y = self.callback(rtMsg)

        if x is None or y is None:
            return
        # 存储新数据点
        self.x_data.append(x)
        self.y_data.append(y)
        
        # 更新曲线的数据
        self.curve.setData(self.x_data, self.y_data)

        # 更新最大值和最小值的显示
        if self.y_data:
            max_y = max(self.y_data)
            min_y = min(self.y_data)
            self.max_text.setText(f'Max: {max_y:.2f}')
            self.min_text.setText(f'Min: {min_y:.2f}')
            self.max_text.setPos(self.x_data[-1], max_y)
            self.min_text.setPos(self.x_data[-1], min_y)

            # 更新y轴的范围
            if self._yRange is None:
                self.plot.setYRange(min_y, max_y, padding=0.1)

            if x >= self.DEFAULT_X_RANGE[1]:
                self.plot.enableAutoRange(axis="x")




class FixedWindow(PlotSubWindow):
    
    def update(self):
        pass



class RollWindow(PlotSubWindow):
    
    DEFAULT_ROLL_DATA_NUM = 1000

    def __init__(self, title, callback, row, col, xRange, yRange, win, rollWindowSize = 10):
        self._moveWindowIndex = None
        self._rollWindowSize = rollWindowSize
        super().__init__(title, callback, row, col, xRange, yRange, win)


    def update(self, rtMsg):

        x, y = self.callback(rtMsg)

        # 存储新数据点
        self.x_data.append(x)
        self.y_data.append(y)
        if x >= self._rollWindowSize:
            if self._moveWindowIndex is None:
                index = self._moveWindowIndex = 0
                # self.plot.autoRange(item=self.plot.getAxis('bottom'))
                self.plot.enableAutoRange(axis="x")
            else:
                index = self._moveWindowIndex = self._moveWindowIndex + 1
            # print(len(self.x_data), index)
            
            if index >= self.DEFAULT_ROLL_DATA_NUM:
                self.x_data = self.x_data[index:]
                self.y_data = self.y_data[index:]
                index = self._moveWindowIndex = 0 

            x_data = self.x_data[index:]
            y_data = self.y_data[index:]
        else:
            x_data = self.x_data
            y_data = self.y_data
        
        # 更新曲线的数据
        self.curve.setData(x_data, y_data)
        # if self._moveWindowIndex is not None:
        #     # self.curve.setPos(self.x_data[self._moveWindowIndex], 0)
        #     print(self._moveWindowIndex, x_data[0])

        # 更新最大值和最小值的显示
        if self.y_data:
            max_y = max(self.y_data)
            min_y = min(self.y_data)
            self.max_text.setText(f'Max: {max_y:.2f}')
            self.min_text.setText(f'Min: {min_y:.2f}')
            self.max_text.setPos(self.x_data[-1], max_y)
            self.min_text.setPos(self.x_data[-1], min_y)

            # 更新y轴的范围
            if self._yRange is None:
                self.plot.setYRange(min_y, max_y, padding=0.1)
            
            # self.lastValue.setText(f"{round(x,5)},{round(y, 5)}")
            # self.lastValue.setPos(self.x_data[self._moveWindowIndex], max_y)



    # def mouseMoved(self, e):
    #     pos = e[0]

    #     if self.plot.sceneBoundingRect().contains(pos):
    #         mousePoint = self.plot.vb.mapSceneToView(pos)
    #         index = int(mousePoint.x())
    #         if index > 0  and index < len(self.x_data):
    #             self._vLine.setPos(mousePoint.x())
    #             self._hLine.setPos(mousePoint.y())
    #             self._mouseValue.setText("1234")



class PlotWindowType(IntEnum):
    ROLL_WINDOW = 0
    COMPRESS_WINDOW = 1
    FIXED_WINDOW = 2


class RealTimePlot(QMainWindow):
    
    def __init__(self, title, msec, suber: Suber):
        super().__init__()
        self.win = pg.GraphicsLayoutWidget(show=True)
        self.setWindowTitle(title)
        self.setCentralWidget(self.win)
        self._plots = {}
        self._suber = suber

        # 在窗口显示时最大化 
        self.showMaximized()
        # 设置一个计时器来调用更新函数
        self._timerInterval = msec
        # self.timer = pg.QtCore.QTimer()
        # self.timer.start(msec)  # 每100毫秒更新一次


    def addSubWindow(self, title, callback, row, col, xRange=None, yRange=None, windowType:PlotWindowType=PlotWindowType.ROLL_WINDOW, rollWindowSize=10):
        if title not in self._plots.keys():
            subWindow = self.createSubWindow(title=title, callback=callback, row=row, col=col, xRange=xRange, yRange=yRange, windowType=windowType,  rollWindowSize=rollWindowSize)
            self._plots.update({title: subWindow})
            self._suber.connect(subWindow.update)
            return subWindow
        else:
            warnings.warn(f"You have add the sub window named {title}")
        

    def delSubWindow(self, title):
        if title in self._plots.keys():
            self._plots.pop(title)
    

    def createSubWindow(self, title, callback, row, col, xRange=None, yRange=None, rollWindowSize: int=10, windowType:PlotWindowType=PlotWindowType.ROLL_WINDOW):
        subWindow = None
        if windowType == PlotWindowType.COMPRESS_WINDOW:
            subWindow = CompressWindow(title=title, callback=callback, row=row, col=col, xRange=xRange, yRange=yRange, win=self.win)
        elif windowType == PlotWindowType.ROLL_WINDOW:
            subWindow = RollWindow(title=title, callback=callback, row=row, col=col, xRange=xRange, yRange=yRange, win=self.win, rollWindowSize=rollWindowSize)
        return subWindow


    def getSubWindow(self, title):
        return self._plots.get(title, None)
    

    def setUpdateTrigger(self, triggerObj:Callable):

        self._triggerObj = triggerObj
        for title, plotWIndow in self._plots.items():
            self._triggerObj.connect(plotWIndow.update)


    def save_all_data(self):
        """保存所有窗口的数据"""
        formatTime = datetime.now().strftime("%Y%m%d_%H_%M_%S")
        for title, plot in self._plots.items():
            filename = f"{formatTime}_{title}_data.csv"
            plot.save_data(filename)

    



if __name__ == '__main__':

    suber = Suber("tcp://127.0.0.1:5556")
    app = QApplication(sys.argv)

    rtPlot = RealTimePlot("Data", 10, suber)
    rtPlot.setUpdateTrigger(suber)
    

    def xFunc(rtMsg:RTMessage):
        return rtMsg.totalTime, float(rtMsg.message.split(",")[0])

    def yFunc(rtMsg: RTMessage):
        return rtMsg.totalTime, float(rtMsg.message.split(",")[1])

    def zFunc(rtMsg: RTMessage):
        z = float(rtMsg.message.split(",")[2])
        if abs(z) <= 0.4:
            z = 0
        return rtMsg.totalTime, z


    rtPlot.addSubWindow(title="X2", callback=xFunc, row=1,col=1, yRange=(-10,10), windowType=PlotWindowType.ROLL_WINDOW)
    rtPlot.addSubWindow(title="Y2", callback=yFunc, row=2,col=1, yRange=(-10,10), windowType=PlotWindowType.ROLL_WINDOW)
    rtPlot.addSubWindow(title="Z2", callback=zFunc, row=3,col=1, yRange=(-2,5), windowType=PlotWindowType.ROLL_WINDOW)


    # 退出时自动保存数据
    def on_exit():
        rtPlot.save_all_data()
        suber.stop()
    # app.aboutToQuit.connect(lambda: rtPlot.save_all_data())
    suber.start()
    rtPlot.show()
    sys.exit(app.exec_())
    suber.stop()
