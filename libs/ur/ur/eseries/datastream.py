#!/usr/bin/env python3
# coding=utf-8
'''
Author       : Jay jay.zhangjunjie@outlook.com
Date         : 2024-07-04 15:37:12
LastEditTime : 2024-09-20 16:12:10
LastEditors  : jay jay.zhangjunjie@outlook.com
Description  : Define Primary And Secondary Port Data Struct
'''





import atexit
import inspect
import socket
from abc import ABC, abstractmethod
from dataclasses import is_dataclass
from queue import Queue
from threading import Thread


# from ur.eseries.datastruct import *
# import ur.eseries.datastruct
# from ur.eseries import datastruct
from . import datastruct


def getAllURDatacls():# -> list:
    # module = importlib.import_module("datastruct")
    dataclasses = []
    for name, obj in inspect.getmembers(datastruct):
        if inspect.isclass(obj) and is_dataclass(obj):
            if hasattr(obj, "fmt"):
                dataclasses.append(obj)
    return dataclasses


URDataClass = getAllURDatacls()




class DataStreamParse(ABC):

    DEFAULT_RECV_MAX_SIZE = 4096
    FREQUENCY = 10                          # hz
    DEFAULT_TIMEOUT = 5


    def __init__(self, ip: str, port: int, autoConnect: bool=True) -> None:
        self.ip, self.PORT = ip, port
        self.connection = False
        self._flowRecvLoop = None
        self.__queue = Queue()
        if autoConnect:
            self.connnect()


    def connnect(self):
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(self.DEFAULT_TIMEOUT)
            self._sock.connect((self.ip, self.PORT))
            atexit.register(self.disconnect)
            self.connection = True
            self._flowRecvLoop = True

             
        except Exception as e:
            print(f"DataFlowParse Connect Failed | IP:{self.ip} Exception:{e}")
            exit()


    def disconnect(self):
        self._flowRecvLoop = False
        self._sock.close()
        self.connection = False


    def flowRecv(self):
        self._recvData = b""
        while self._flowRecvLoop:
            recvData = self._sock.recv(self.DEFAULT_RECV_MAX_SIZE)
            self._recvData += recvData
            while True:
                oneFrameData, self._recvData  = self.fetchOneFrameData(self._recvData)
                if oneFrameData is not None:
                    self.__queue.put(oneFrameData)
                else:
                    break


    def parseFlowData(self):
        while self._flowRecvLoop:
            if not self.__queue.empty():
                data = self.__queue.get()
                self.parseOneFrameData(data)


    @abstractmethod
    def parseOneFrameData(self, oneFrameData: bytes):
        raise NotImplementedError


    @abstractmethod
    def fetchOneFrameData(self, dataFlow: bytes):
        raise NotImplementedError


    def monitorStart(self):
        if not self.connection:
            self.connnect()
        self._flowRecvLoop = True
        self.flowRecvThread = Thread(target=self.flowRecv, name=f"UR DataFlowParse | IP:{self.ip}, Port:{self.PORT}", daemon=True)
        self.flowRecvThread.start()
        self.parseFlowRecvThread = Thread(target=self.parseFlowData, name=f"UR ParseDataFlowParse | IP:{self.ip}, Port:{self.PORT}", daemon=True)
        self.parseFlowRecvThread.start()


    def monitorStop(self):
        self._flowRecvLoop = False
        self.disconnect()
        self.__queue.queue.clear()






# if __name__ == "__main__":


#     # primary = PrimaryPortStreamParse(ip="192.168.1.103" ,autoConnect=True)
#     # primary.monitorStart()

#     secondary = SecondaryPortStreamParse(ip="192.168.10.109", autoConnect=True)
#     secondary.monitorStart()

#     primary   = PrimaryPortStreamParse(ip="192.168.10.109", autoConnect=True)
#     primary.monitorStart()
#     time.sleep(1)
#     while 1:
#         time.sleep(5)
#         print(secondary.CartesianInfo.Z)
#         print(primary.KeyMessage)
        # print(secondary.RobotModeData            )
        # print(secondary.JointData                )
        # print(secondary.ToolData                 )
        # print(secondary.ToolData                 )
        # print(secondary.MasterboardData1         )
        # print(secondary.MasterboardData2         )
        # print(secondary.CartesianInfo            )
        # print(secondary.KinematicsInfo           )
        # print(secondary.ConfigurationData        )
        # print(secondary.ForceModeData            )
        # print(secondary.AdditionInfo             )
        # print(secondary.CalibrationData          )
        # print(secondary.ToolCommunicationInfo    )
        # print(secondary.ToolModeInfo             )

