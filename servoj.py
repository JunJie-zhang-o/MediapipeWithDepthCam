#!/usr/bin/env python3
# coding=utf-8
'''
Author       : Jay jay.zhangjunjie@outlook.com
Date         : 2024-07-24 13:44:35
LastEditTime : 2024-09-20 16:15:19
LastEditors  : jay jay.zhangjunjie@outlook.com
Description  : Orbbec+Mediapipe+摇操作
'''


"""
    MaxX:200mm   MinX:-90mm
    MaxY:-250mm   MinY:-450mm
    MaxZ:100mm   MinZ:20mm

"""

import sys,pathlib


import zmq

from  servoj_utils.fliter import BandLimiter, DValueLimiter
from servoj_utils.smoother import MovingAverage3D

sys.path.append(f"{pathlib.Path(__file__).parent.parent.parent}")

from threading import Event, Thread
import time
from libs.ur.ur.eseries import URERobot, ConfigFile


# import numpy as np
# np.set_printoptions(suppress=True, threshold=np.nan)


# ----------------utils-------------
def setp_to_list(setp):
    temp = []
    for i in range(0, 6):
        temp.append(setp.__dict__["input_double_register_%i" % i])
    return temp


def list_to_setp(setp, list):
    for i in range(0, 6):
        setp.__dict__["input_double_register_%i" % i] = round(list[i],6)
    return setp
# ----------------utils-------------

class Suber(Thread):

    def __init__(self, address, topic=""):
        super().__init__(name=f"Suber Thread | address:{address}, topic:{topic}", daemon=True)
        self.address = address
        self.topic = topic
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(self.address)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)
        self._stop_event = Event()
        self.message = "0,0,0"
        self.internal = 0
        self.timestamp = 0


    def run(self):
        print(f"Subscriber started, listening to {self.topic} on {self.address}")
        t1 = time.time()
        while not self._stop_event.is_set():
            try:
                message = self.socket.recv_string(flags=zmq.NOBLOCK)
                t2 = time.time()
                self.internal = t2 - t1
                self.timestamp += self.internal
                t1 = t2
                print(f"Received message on topic {self.topic}: {message} | timestamp:{self.timestamp} | internal:{self.internal}")
                self.message = message
            except zmq.Again:
                pass


    def stop(self):
        self._stop_event.set()
        self.socket.close()
        self.context.term()
        print("Subscriber stopped")




# -------------logic control-----------------

class Servoj():


    def __init__(self, ip:str) -> None:
        
        self.robot = URERobot(ip, autoConnect=True)

        self.rtde = self.robot.createRTDEInterface()

        conf = ConfigFile('libs/ur/case/servoj/rtde_data_example.xml')
        self.state_names, self.state_types = conf.get_recipe('state')  # Define recipe for access to robot output ex. joints,tcp etc.
        self.setp_names, self.setp_types = conf.get_recipe('setp')  # Define recipe for access to robot input
        self.watchdog_names, self.watchdog_types= conf.get_recipe('watchdog')

        self._initRTDEParams()


    def _initRTDEParams(self):
        # setup rtde params
        FREQUENCY = 125  # send data in 500 Hz instead of default 125Hz
        self.rtde.send_output_setup(self.state_names, self.state_types, FREQUENCY)
        self.setp = self.rtde.send_input_setup(self.setp_names, self.setp_types)  # Configure an input package that the external application will send to the robot controller
        self.watchdog = self.rtde.send_input_setup(self.watchdog_names, self.watchdog_types)
        self.rtde.send_start()

        recvLoopThread = Thread(target=self.rtdeThread, daemon=True, name="RtdeRecvLoopThread")
        recvLoopThread.start()


    def rtdeThread(self):
        while True:
            self.rtdeRecvState = self.rtde.receive()
            # self.rtde.send(self.watchdog)
            # self.rtde.send(self.setp)


    def initRobotState(self, initPose) -> list:
        list_to_setp(self.setp, initPose)  # changing initial pose to setp
        self.setp.input_bit_registers0_to_31 = 0
        self.rtde.send(self.setp) # sending initial pose

        self.watchdog.input_int_register_0 = 0

        self.rtde.send(self.watchdog)
        self.rtde.send(self.setp)
        while 1:
            print('Boolean 1 is False, please click CONTINUE on the Polyscope')
            if hasattr(self, "rtdeRecvState"):
                if self.rtdeRecvState.output_bit_registers0_to_31 == True:
                    print('Boolean 1 is True, Robot Program can proceed to mode 1\n')
                    break
            else:
                print("wait redeRecvThread Start!!!")

        self.watchdog.input_int_register_0 = 1
        while 1:
            time.sleep(0.005)
            print('Waiting for movej() to finish')
            self.rtde.send(self.watchdog)
            if self.rtdeRecvState.output_bit_registers0_to_31 == False:
                print('Proceeding to mode 2\n')
                time.sleep(0.05)
                # self.addPoseToServoj([0,0,0,0,0,0])
                self.watchdog.input_int_register_0 = 2
                self.rtde.send(self.watchdog)
                return self.rtdeRecvState.actual_TCP_pose


    def addPoseToServoj(self, pose):
        list_to_setp(self.setp, pose)
        self.rtde.send(self.setp)



# -------------------------------------------------------




# -----------------------------------------------------------

class Filter:
    
    def __init__(self, x, y, z) -> None:
        
        self._x, self._y, self._z = x, y, z
        self.average = MovingAverage3D(windowsSize=5)

        # 数据波动限制
        self.xLimiter = DValueLimiter(maxDiffThreshold=20)
        self.yLimiter = DValueLimiter(maxDiffThreshold=20)
        self.zLimiter = DValueLimiter(maxDiffThreshold=20)
        # 区域保护
        self.xAreaLimiter = BandLimiter(highThreshold=0.3, lowThreshold=-0.04)
        self.yAreaLimiter = BandLimiter(highThreshold=-0.36, lowThreshold=-0.69)
        self.zAreaLimiter = BandLimiter(highThreshold=0.3, lowThreshold=0.06)

        self._dbFlag = False
    
    def fliter(self, x, y, z):
        # 做一次平滑
        result = self.average.handle(x, y, z)
        if result is not None:
            (x, y, z) = result
            # 滤掉跳动比较大的数据
            x = self.xLimiter.limit(x)
            y = self.yLimiter.limit(y)
            z = self.zLimiter.limit(z)
            _x = x + self._x
            _y = y + self._y
            _z = z + self._z
            # 区域限制
            _x = self.xAreaLimiter.limit(_x)
            _y = self.yAreaLimiter.limit(_y)
            _z = self.zAreaLimiter.limit(_z)
            if self._dbFlag:
                self._x, self._y, self._z = _x, _y, _z
                self._dbFlag = False
            return _x, _y, _z

    def setDBFlag(self):
        self._dbFlag = True



# -------------------------------------------------------
if __name__ == "__main__":


    # ip = "192.168.40.127"  # Sim
    ip = "192.168.1.103"  # Real
    # robot = URERobot(ip,True)
    # dash = robot.createDashboardInterface()
    # primaryMonitor = robot.createPrimaryMonitorInterface()
    # primaryMonitor.monitorStart()
    # time.sleep(1)
    # print(primaryMonitor.CartesianInfo)
    # while 1:
    #     if primaryMonitor.CartesianInfo is not None:
    #         x = primaryMonitor.CartesianInfo.X 
    #         y = primaryMonitor.CartesianInfo.Y 
    #         z = primaryMonitor.CartesianInfo.Z 
    #         rx = primaryMonitor.CartesianInfo.Rx 
    #         ry = primaryMonitor.CartesianInfo.Ry 
    #         rz = primaryMonitor.CartesianInfo.Rz 
    #         HOME_POSE = [x, y, z, rx, ry, rz]
    #         print(HOME_POSE)
    #         break
    #     time.sleep(0.1)

    # time.sleep(2)
    # exit()

    # suber = Suber("tcp://192.168.1.2:5556")
    # suber = Suber("tcp://192.168.40.241:5556")
    # suber = Suber("tcp://192.168.1.7:5556")

    puber = zmq.Context().socket(zmq.PUB)
    puber.bind("tcp://*:5557")
    puber.set_hwm(100)

    
    suber = Suber("tcp://127.0.0.1:5556")

    suber.start()

    # app = QApplication(sys.argv)
    # rtPlot = RealTimePlot("数据", 100)

    # HOME_POSE = [0.01982, -0.36266, 0.09777, 3.141, -0.052, 0]
    HOME_POSE = [0.0833, -0.46700, 0.10617, -2.2214, -2.2214, 0]
    # HOME_POSE[0], HOME_POSE[1], HOME_POSE[2] = 0,0,0
    filter = Filter(HOME_POSE[0], HOME_POSE[1], HOME_POSE[2])
    filter1 = Filter(HOME_POSE[0], HOME_POSE[1], HOME_POSE[2])
    filter2 = Filter(HOME_POSE[0], HOME_POSE[1], HOME_POSE[2])

    from xmlrpc.server import SimpleXMLRPCServer
    def setData():
        filter.setDBFlag()
        filter1.setDBFlag()
        filter2.setDBFlag()
    server = SimpleXMLRPCServer(("0.0.0.0", 9120), allow_none=True)
    server.register_function(setData)
    Thread(target=server.serve_forever, daemon=True, name="XMLRPC Server").start()
    
    # rtPlot.addSubWindow("X", lambda: (float(suber.timestamp), filter.fliter(float(suber.message.split(",")[0]), 0, 0)[0]), 1,1)
    # rtPlot.addSubWindow("Y", lambda: (float(suber.timestamp), filter1.fliter(float(suber.message.split(",")[1]),0,0)[0]), 2,1)
    # rtPlot.addSubWindow("Z", lambda: (float(suber.timestamp), filter2.fliter(float(suber.message.split(",")[2]),0,0)[0]), 3,1)
    # rtPlot.show()
    # sys.exit(app.exec_())
    sj = Servoj(ip=ip)
    initPose = sj.initRobotState(HOME_POSE)

    sp = [i for i in HOME_POSE]
    flag = 0.012


    while 1:
        t1 = time.time()
        values = suber.message

        if values is not None:

            [trsfX, trsfY, trsfZ] = values.split(",")

            dx = float(trsfZ) * 0.003
            dy = float(trsfX) * 0.003
            dz = float(trsfY) * 0.003  # 1

            ret = filter.fliter(dx, dy, dz)
            if ret is not None:
                x, y, z = ret

                sp[0] = x
                sp[1] = y
                sp[2] = z

                # sp = [float(trsfX)*flag, float(trsfY)*flag, 0, 0, 0, 0]
                # sp = [float(trsfX)*flag, float(trsfZ)*0.1, float(trsfY)*-flag, 0, 0, 0]
                print(sp)
                # diff = [f"{sp[i]-HOME_POSE[i]:f}" for i in range(6)]
                # print(diff)
                sj.addPoseToServoj(sp)
                puber.send_string(f"{x},{y},{z}") 



        time.sleep(0.03)
    # sys.exit(app.exec_())







