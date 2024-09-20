#!/usr/bin/env python3
# coding=utf-8
'''
Author       : Jay jay.zhangjunjie@outlook.com
Date         : 2024-07-07 20:56:17
LastEditTime : 2024-07-22 14:35:46
LastEditors  : Jay jay.zhangjunjie@outlook.com
Description  : 
'''


import socket
import struct

from loguru import logger

from ur.eseries.datastream import DataStreamParse, URDataClass
from ur.eseries.datastruct import *


class Primary:

    PORT = 30001
    DEFAULT_TIMEOUT = 3
    SCRIPT_HEADER = "def"
    SCRIPT_FOOTER = "end"


    def __init__(self, ip: str, autoConnect: bool=False) -> None:
        self.ip = ip

        if autoConnect:
            self.conncet()


    def conncet(self):
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(self.DEFAULT_TIMEOUT)
            self._sock.connect((self.ip, self.PORT))
        except Exception as e:
            print(f"UR Primary Port Connect Failed | IP:{self.ip} Port:{self.PORT} Exception:{e}")
            exit()

    
    def disconnect(self):
        self._sock.close()
        self.connection = False


    def _getFullScript(self, script: str, scriptName: str="pyURScript") -> str:

        fullScript = f"{self.SCRIPT_HEADER} {scriptName}():\n"

        for line in script.split("\n"):
            fullScript += f"  {line}\n"
        fullScript += f"{self.SCRIPT_FOOTER}\n"
        return fullScript


    def sendSingleScript(self, script: str):
        # self._sock.sendall(self._getFullScript(str(script)).encode())
        rawScript = self._getFullScript(str(script))
        self.sendRawScript(rawScript)


    def sendRawScript(self, script: str):
        print(script)
        self._sock.send(str(script).encode())




class PrimaryMonitor(DataStreamParse):

    PORT = 30011

    def __init__(self, ip: str, autoConnect: bool = True) -> None:
        super().__init__(ip, self.PORT, autoConnect)

        # RobotState
        self.RobotModeData            : RobotModeData         | None = None
        self.JointData                : RobotModeData         | None = None
        self.ToolData                 : RobotModeData         | None = None
        self.ToolData                 : RobotModeData         | None = None
        self.MasterboardData1         : MasterboardData1      | None = None
        self.MasterboardData2         : MasterboardData2      | None = None
        self.CartesianInfo            : CartesianInfo         | None = None
        self.KinematicsInfo           : KinematicsInfo        | None = None
        self.ConfigurationData        : ConfigurationData     | None = None
        self.ForceModeData            : ForceModeData         | None = None
        self.AdditionInfo             : AdditionInfo          | None = None
        self.CalibrationData          : CalibrationData       | None = None
        self.ToolCommunicationInfo    : ToolCommunicationInfo | None = None
        self.ToolModeInfo             : ToolModeInfo          | None = None

        # RobotMessage
        self.VersionMessage: VersionMessage | None = None
        self.SafetyModeMessage: SafetyModeMessage | None = None
        self.RobotCommMessage: RobotCommMessage | None = None
        self.KeyMessage: KeyMessage | None = None
        self.ProgramThreadsMessage: ProgramThreadsMessage | None = None
        self.PopupMessage: PopupMessage | None = None
        self.RequestValueMessage: RequestValueMessage | None = None
        self.TextMessage: TextMessage | None = None
        self.RuntimeExceptionMessage: RuntimeExceptionMessage | None = None
        self.GlobalVariablesSetupMessage: GlobalVariablesSetupMessage | None = None
        self.GlobalVariablesUpdateMessage: GlobalVariablesUpdateMessage | None = None


    def _parseTwoUnknowNumMsg(self, cls, msg):
        cls.charNum1, cls.charNum2 = 0, 0
        cls._refreshFmt()
        index = cls.fmt.find(str(cls.charNum1))
        cls.charNum1 =  struct.unpack(f"!{cls.fmt[:index]}", msg[:struct.calcsize(f"!{cls.fmt[:index]}")])[index - 1]
        cls._refreshFmt()
        cls.charNum2 = len(msg) - struct.calcsize(f"!{cls.fmt}")
        cls._refreshFmt()
        return struct.unpack(f"!{cls.fmt}", msg)

    
    def _parseOneUnkonwNumMsg(self, cls, msg):
        cls.charNum = 0
        cls._refreshFmt()
        cls.charNum = len(msg) - struct.calcsize(f"!{cls.fmt}")
        cls._refreshFmt()
        return struct.unpack(f"!{cls.fmt}", msg)
    

    def _parseProgramThreadMessage(self, msg):
        headerLength = struct.calcsize("!iBQbB")
        [size, msgtype, timestamp, source, robotMessageType] =  struct.unpack("!iBQbB", msg[:headerLength])
        threads = []
        programData = msg[headerLength:]
        while True:
            [labelID, labelNameLength] = struct.unpack("!ii", programData[:8])
            threadNameLength = struct.unpack("!i", programData[8+labelNameLength:12+labelNameLength])[0]
            threads.append(ProgramThread(*struct.unpack(f"!ii{labelNameLength}si{threadNameLength}s", programData[:12+labelNameLength+threadNameLength])))
            programData = programData[12+labelNameLength+threadNameLength:]
            print('')
            if len(programData) == 0:
                return [size, msgtype, timestamp, source, robotMessageType, threads]



    def parseOneFrameData(self, oneFrameData: bytes):
        # print("------------------------------------------------------")
        while 1:
            length =  struct.unpack("!i", oneFrameData[:4])[0]
            type   =  struct.unpack("!b", oneFrameData[4:5])[0]

            if len(oneFrameData) >= length:
                parseData = oneFrameData[:length]   # 取出一个数据包数据
                data = oneFrameData[length:]
            else:
                print("长度也不够")
                exit()
            # print(f"P--- Length:{length} Type:{type}")
            if type == 16:
                oneFrameData = oneFrameData[5:]
                continue
            else:
                oneFrameData = oneFrameData[length:]

            for c in URDataClass:
                # 由于MasterboardData数据分为两种情况,需要额外分开处理
                # print(c)
                if c.code == type == 3:
                    l = struct.calcsize(f"!{c.fmt}")    # 计算标准数据包的长度
                    if length != l:
                        continue

                if c.code == type == 20:
                    [size, _, _, source, robotMessageType] =  struct.unpack("!iBQbB", parseData[:struct.calcsize("!iBQbB")])
                    # print(robotMessageType)
                    # print(c, robotMessageType)
                    if robotMessageType == VersionMessage.robotMsgType == c.robotMsgType:
                        arg = self._parseTwoUnknowNumMsg(VersionMessage, parseData)
                        logger.info(f"{VersionMessage(*arg)}")
                        setattr(self, c.bindProperty, VersionMessage(*arg))
                        break
                    elif robotMessageType == KeyMessage.robotMsgType == c.robotMsgType:
                        arg = self._parseTwoUnknowNumMsg(KeyMessage, parseData)
                        logger.debug(f"P | {KeyMessage(*arg)}")
                        setattr(self, c.bindProperty, KeyMessage(*arg))
                        break

                    elif robotMessageType == PopupMessage.robotMsgType == c.robotMsgType:
                        arg = self._parseOneUnkonwNumMsg(PopupMessage, parseData)
                        logger.error(f"P | {PopupMessage(*arg)}")
                        setattr(self, c.bindProperty, PopupMessage(*arg))
                        break

                    elif robotMessageType == RequestValueMessage.robotMsgType == c.robotMsgType:
                        arg = self._parseOneUnkonwNumMsg(RequestValueMessage, parseData)
                        logger.error(f"P | {RequestValueMessage(*arg)}")
                        setattr(self, c.bindProperty, RequestValueMessage(*arg))
                        break

                    elif robotMessageType == TextMessage.robotMsgType == c.robotMsgType:
                        arg = self._parseOneUnkonwNumMsg(TextMessage, parseData)
                        logger.error(f"P | {TextMessage(*arg)}")
                        setattr(self, c.bindProperty, TextMessage(*arg))
                        break

                    elif robotMessageType == RobotCommMessage.robotMsgType == c.robotMsgType:
                        arg = self._parseOneUnkonwNumMsg(RobotCommMessage, parseData)
                        logger.error(f"P | {RobotCommMessage(*arg)}")
                        setattr(self, c.bindProperty, RobotCommMessage(*arg))
                        break

                    elif robotMessageType == c.robotMsgType and robotMessageType != 14:
                        arg = struct.unpack(f"!{c.fmt}", parseData)
                        logger.critical(f"P | {c(*arg)}")
                        setattr(self, c.bindProperty, c(*arg))
                        break
                    elif robotMessageType == c.robotMsgType and robotMessageType == 14:
                        # print(parseData)
                        arg = self._parseProgramThreadMessage(parseData)
                        logger.critical(f"P | {c(*arg)}")
                        setattr(self, c.bindProperty, c(*arg))
                        break
                    else:
                        continue

                if c.code == type:
                    l = struct.calcsize(f"!{c.fmt}")    # 计算标准数据包的长度

                    if length < l:
                        break
                    if l >= length:
                        args = struct.unpack("!"+c.fmt, parseData[:length]) # 解析所有的数据包数据

                        params = c.__annotations__  # 获取dataclass的属性
                        initArgs = []
                        n = 0
                        # 按照实际类的参数进行调整,主要是兼顾list的情况
                        for param in params:
                            if params[param].__name__ == "ClassVar":
                                continue
                            if  params[param] is JointsFloat or params[param] is JointsInt:     # 组合6个关节的参数
                                initArgs.append(args[n:n+6])
                                n = n + 6
                                continue
                            initArgs.append(args[n])
                            n += 1
                        # 按照实际类的参数进行调整,主要是兼顾list的情况
                        ins = c(*initArgs)
                        # print(f"P | {ins}")
                        setattr(self, c.bindProperty, ins)
                        break

            
            if len(data) == 0:
                break


    def fetchOneFrameData(self, dataStream: bytes):
        while True:
            if len(dataStream) > struct.calcsize("!iB"):
                [dataLength, dataType ]=  struct.unpack("!iB", dataStream[:5])

                # print(dataLength, dataType)
                if len(dataStream) >= dataLength:   # 数据包长度大于要解析的长度
                    data = dataStream[:dataLength]
                    return data, dataStream[dataLength:]
                else:   # 不满足一个数据包的长度,不解析
                    return None, dataStream
            else:       # 无法获取数据包长度和类型,不解析
                return None, dataStream

