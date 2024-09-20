






import struct

from ur.eseries.datastream import DataStreamParse, URDataClass
from ur.eseries.datastruct import *
from ur.eseries.primary import Primary


class Secondary(Primary):

    PORT = 30002

    SCRIPT_HEADER = "sec"




class SecondaryMonitor(DataStreamParse):

    PORT = 30012

    def __init__(self, ip: str, autoConnect: bool = True) -> None:
        super().__init__(ip, self.PORT, autoConnect)

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
            # print(f"--- Length:{length} Type:{type}")
            if type == 16:
                oneFrameData = oneFrameData[5:]
                continue
            else:
                oneFrameData = oneFrameData[length:]

            for c in URDataClass:
                # 由于MasterboardData数据分为两种情况,需要额外分开处理
                if c.code == 3:
                    l = struct.calcsize(f"!{c.fmt}")    # 计算标准数据包的长度
                    if length != l:
                        continue

                if c.code == type:
                    l = struct.calcsize(f"!{c.fmt}")    # 计算标准数据包的长度

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
                        # print(ins)
                        setattr(self, c.bindProperty, ins)
                        break

            
            if len(data) == 0:
                break


    def fetchOneFrameData(self, dataStream: bytes):
        while True:
            if len(dataStream) > struct.calcsize("!iB"):
                dataLength =  struct.unpack("!i", dataStream[:4])[0]
                dataType   =  struct.unpack("!B", dataStream[4:5])[0]

                # print(dataLength, dataType)
                if len(dataStream) >= dataLength:   # 数据包长度大于要解析的长度
                    if dataType == 16:              # 解析RobotState数据
                        if len(dataStream) >= dataLength:
                            data = dataStream[:dataLength]
                            return data, dataStream[dataLength:]
                        else:
                            print("长度不够")
                            return None, dataStream            
                    else:                           # 非RobotState数据不解析
                        dataStream = dataStream[dataLength:]
                else:   # 不满足一个数据包的长度,不解析
                    return None, dataStream
            else:       # 无法获取数据包长度和类型,不解析
                return None, dataStream

