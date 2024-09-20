#!/usr/bin/env python3
# coding=utf-8
'''
Author       : Jay jay.zhangjunjie@outlook.com
Date         : 2024-07-24 13:44:35
LastEditTime : 2024-08-07 11:12:55
LastEditors  : Jay jay.zhangjunjie@outlook.com
Description  : 触觉反向控制机器人-移动回弹
'''

import os
import pathlib
import sys

import zmq

sys.path.append(f"{pathlib.Path(__file__).parent.parent.parent}")

import time
from threading import Event, Thread

from ur.eseries import RTDE, ConfigFile, URERobot

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


# -------------logic control-----------------

class Servoj():


    def __init__(self, ip:str) -> None:
        
        self.robot = URERobot(ip, autoConnect=True)

        self.rtde = self.robot.createRTDEInterface()

        conf = ConfigFile('./examples/rtde_data_example.xml')
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

HOME_POSE = [0.12463821364715821, -0.21147695906075042, 0.13790726529380445, -2.920674488647763, 1.146963650993509, 0.019735537141191365]


# -----------------------------------------------------------


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
        self.message = None


    def run(self):
        print(f"Subscriber started, listening to {self.topic} on {self.address}")
        while not self._stop_event.is_set():
            try:
                message = self.socket.recv_string(flags=zmq.NOBLOCK)
                print(f"Received message on topic {self.topic}: {message}")
                self.message = message
            except zmq.Again:
                pass


    def stop(self):
        self._stop_event.set()
        self.socket.close()
        self.context.term()
        print("Subscriber stopped")

# import zmq
# context = zmq.Context()
# suber = context.socket(zmq.SUB)
# suber.connect("tcp://127.0.0.1:5556")
# suber.setsockopt_string(zmq.SUBSCRIBE, "")
# suber = Suber("tcp://127.0.0.1:5556")
suber = Suber("tcp://192.168.1.2:5556")

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


    sj = Servoj(ip=ip)
    # initPose = sj.initRobotState(HOME_POSE)
    initPose = sj.initRobotState([0,0,0,0,0,0])

    input()
    # pose = initPose.copy()
    sp = [i for i in HOME_POSE]
    flag = 0.018
    suber.start()
    while 1:
        t1 = time.time()
        values = suber.message
        # print(values)
        # continue
        if values is not None:
            # print(time.time() - t1)
            # continue
            # print(values)
            # values = "0,0,1"
            [trsfX, trsfY, trsfZ] = values.split(",")
            
            # sp[0] = HOME_POSE[0] + float(trsfX) * flag
            # sp[1] = sp[1] + float(trsfY) * flag
            # sp[2] = sp[2] + float(trsfZ) * flag

            # 鸡蛋反向位移
            # sp = [float(trsfX)*flag, float(trsfY)*flag, 0, 0, 0, 0]
            sp = [float(trsfX)*flag, 0, float(trsfY)*-flag, 0, 0, 0]
            print(sp)
            # diff = [f"{sp[i]-HOME_POSE[i]:f}" for i in range(6)]
            # print(diff)
            sj.addPoseToServoj(sp)

            # if sp[2] >= 0.4:
            #     flag = -0.001
            # if sp[2] <= 0.3:
            #     flag = 0.001

        time.sleep(0.008)







