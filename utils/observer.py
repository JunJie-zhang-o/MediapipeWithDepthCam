


from abc import ABC, abstractmethod
from copy import deepcopy
import logging
from threading import Thread
import time
from typing import Dict
from enum import Enum
from xmlrpc import client

import zmq

from utils.landmark import GestureLandMarkDetector, GestureRecognizerResultData
from utils.logger import RemoteTCPServerLogHandler


logger = logging.getLogger('socket_logger')
# logger.addHandler(RemoteTCPServerLogHandler("192.168.40.216"))
logger.setLevel(logging.DEBUG)

class Observer(ABC):

    @abstractmethod
    def updata(self, obj):
        pass


class Monitor(Thread):


    def __init__(self, getDataCallback) -> None:
        self._getValue = getDataCallback
        self._loop = True


    def stop(self):
        self._loop = False

    def run(self) -> None:
        
        while self._loop:
            pass

class ActionTrigger:
    """
        根据持续的时间动作的触发器
    """

    def __init__(self, name) -> None:
        self.name = name
        self._callbacks = {}
        self._duration = {}
        self._data = {}
        self._volatuationData = {}
        self._t = {}

    
    def addCallback(self, callback, duration, volatuationData=None, flag=None):
        if flag is None:
            flag = str(len(self._callbacks))
        if flag not in self._callbacks.keys():
            self._callbacks.update({flag: callback})
            self._duration.update({flag: duration})
            self._volatuationData.update({flag: volatuationData})
        else:
            raise Exception(f"The flag already exists in ActionTrigger:{self.name}")


    def popCallback(self, flag=None):
        if flag is None:
            flag = str(len(self._callbacks) - 1)
        self._callbacks.pop(flag)
        self._duration.pop(flag)
        self._volatuationData.pop(flag)


    def trigger(self, t:float, v:float, diffIsPositiveValue: bool=True):
        trigger_result = []
        for k in self._callbacks.keys():

            callback = self._callbacks.get(k)
            duration = self._duration.get(k)
            volatuationData = self._volatuationData.get(k)
            # 先判断数值是否波动,波动范围内不需要处理
            data = self._data.get(k, None)
            if self.name == GestureObserver.FuncNameLists.INCREASE.value or self.name == GestureObserver.FuncNameLists.REDUCE.value:
                logger.critical(f"Name:{self.name}, duration:{duration}, volatuationData:{volatuationData} | t:{t}, v:{v}, diffIsPositiveValue:{diffIsPositiveValue}")
            if self.name == GestureObserver.FuncNameLists.INCREASE.value:
                # print(self.name, data)
                pass
            if data is not None and v is not None and volatuationData is not None:
                diff = v - data
                if abs(diff) <= volatuationData: 
                    # self._t[k] = None
                    if self.name == GestureObserver.FuncNameLists.INCREASE.value or self.name == GestureObserver.FuncNameLists.REDUCE.value:
                        logger.debug(f"{self.name} diff <= volatuationData | diff:{diff} data:{data}")
                    continue                   
                # 绝对值只能判断是否波动,加上差值的判断
                if diffIsPositiveValue:
                    if diff < 0: 
                        if self.name == GestureObserver.FuncNameLists.INCREASE.value or self.name == GestureObserver.FuncNameLists.REDUCE.value:
                            logger.debug(f"{self.name} diff < 0 and diffIsPositiveValue > 0")
                        continue
                if not diffIsPositiveValue:
                    if diff > 0: 
                        if self.name == GestureObserver.FuncNameLists.INCREASE.value or self.name == GestureObserver.FuncNameLists.REDUCE.value:
                            logger.debug(f"{self.name} diff > 0 and diffIsPositiveValue < 0")
                        continue

            
            # 持续时间内,记录首次触发的时间和数据
            dbT = self._t.get(k, None)
            if dbT is None:
                self._t.update({k: t})
                if v is not None:
                    self._data.update({k: v})
                if self.name == GestureObserver.FuncNameLists.INCREASE.value or self.name == GestureObserver.FuncNameLists.REDUCE.value:
                    logger.info(f"{self.name} record | t:{t} v:{v}")
                continue
            
            # 时间和数据都满足,判断是否为持续触发和满足时间周期
            if t - dbT >= duration:
                callback()
                if self.name == GestureObserver.FuncNameLists.INCREASE.value:
                    # print(self.name, t, dbT, v, data)
                    pass
                if self.name == GestureObserver.FuncNameLists.REDUCE.value:
                    # print(self.name, t, dbT, v, data)
                    pass
                self._t.update({k: t})
                self._data.update({k: v})
                trigger_result.append(True)
                if self.name == GestureObserver.FuncNameLists.INCREASE.value or self.name == GestureObserver.FuncNameLists.REDUCE.value:
                    logger.info(f"{self.name} callback | t:{t} v:{v} result:{True}")
        return trigger_result
                

    def reset(self):
        if self.name == GestureObserver.FuncNameLists.INCREASE.value or self.name == GestureObserver.FuncNameLists.REDUCE.value:
            logger.debug(f"name:{self.name} reset")
        for k in self._callbacks.keys():
            self._t.update({k: None})
            self._data.update({k: None})




class GestureObserver(Observer):
    """
        手势特征观察者
    """
    class FuncNameLists(Enum):
        UNKONWN       = "UnKonwn"               # 未知手势
        OPEN          = "Open_Palm"             # 打开手掌
        CLOSE         = "Closed_Fist"           # 关闭手掌
        POINTING_UP   = "Pointing_Up"           # 食指指天
        THUMB_DOWN    = "Thumb_Down"            # 大拇指向上
        THUMB_UP      = "Thumb_Up"              # 大拇指向下
        VICTORY       = "Victory"               # yeah
        LOVE          = "ILoveYou"              # 1、3、5手指向上,手掌面向或手背面向相机都可以
        INCREASE      = "Increase"              # 大拇指和食指间距拉长
        REDUCE        = "Reduce"                # 大拇指和食指间距减小

    def __init__(self, cam) -> None:
        
        # self.duration = {}
        # self.volatuationData = {}
        self._callbackFuncDict:Dict[str, ActionTrigger] = {}

        self.__t = {}
        self._db = {}
        self._cam = cam
        self.wristPose = None
        self._cali_flag = False  
        self._record_start_pose = None
        self._last_diff = None
        self.puber = zmq.Context().socket(zmq.PUB)
        self.puber.bind("tcp://*:5556")
        self.puber.set_hwm(100)

    
    def register_callback(self, name:FuncNameLists, callback, duration, volatuationData=None):
        if name in self.FuncNameLists:    
            if name.value in self._callbackFuncDict.keys():
                # self._callbackFuncDict[name.value].append(callback)
                self._callbackFuncDict.get(name.value).addCallback(callback, duration, volatuationData)
                # self.duration[name.value].append(duration)
                # self.volatuationData[name.value].append(volatuationData)
            else:
                self._callbackFuncDict[name.value] = ActionTrigger(name.value)
                self._callbackFuncDict.get(name.value).addCallback(callback, duration, volatuationData)
                # self.duration[name.value] = [duration]
                # self.volatuationData[name.value] = [volatuationData]
        else:
            print("name not in funcNameLists")
            exit(0)


    def deregister_callback(self, name):
        self._callbackFuncDict.pop(name)

    
    def updata(self, obj:GestureLandMarkDetector):
        # t = 0
        # while 1:
        #     time.sleep(0.001)
            now = time.time()
            if type(obj.result) is not GestureRecognizerResultData:
                # continue
                return 
            gesture = obj.result.gestures.category_name
           
            # print(gesture)
            dis = obj.get_thumb_indexfinger_tip_dis()


            # if gesture in self._callbackFuncDict.keys():
            for name, trigger in self._callbackFuncDict.items():
                trigger:ActionTrigger
                if gesture == name:
                    trigger.trigger(now, dis)
                elif name == self.FuncNameLists.INCREASE.value:
                    pass
                elif name == self.FuncNameLists.REDUCE.value:
                    pass
                else:
                    trigger.reset()

            # 距离检测
            if self.FuncNameLists.INCREASE.value in self._callbackFuncDict.keys() or self.FuncNameLists.REDUCE.value in self._callbackFuncDict.keys():
                if self._db.get("dis", None) is None:
                    self._db["dis"] = dis
                    # continue 
                    logger.debug("db dis is None")
                    return 
                # db_dis = self._db.get("dis")
                # diff = dis - db_dis
                # if diff > 0:
                # 打开和关闭的处理
                ret = self._callbackFuncDict.get(self.FuncNameLists.INCREASE.value).trigger(now, dis, diffIsPositiveValue=True)
                if True in ret:
                    self._callbackFuncDict.get(self.FuncNameLists.REDUCE.value).reset()
                # if diff < 0:
                ret = self._callbackFuncDict.get(self.FuncNameLists.REDUCE.value).trigger(now, dis, diffIsPositiveValue=False)
                if True in ret:
                    self._callbackFuncDict.get(self.FuncNameLists.INCREASE.value).reset()
                # t = t+1
                # if t == 100:
                #     self._db["dis"] = dis
                #     t = 0
            if self._cali_flag == False:
                self.puber.send_string(f"0,0,0") 
                self._last_diff = None
                return 
            
            # 检测手腕数据
            wristPose =  obj.result.hand_landmarks.wrist.getPose()
            # print(wristPose)
            if obj.output_image is not None:
                imgH, imgW = obj.output_image.shape[:2]
                # todo暂时不做是否在屏幕外的检测
                wristXY = [int(wristPose[0]*imgW), int(wristPose[1]*imgH)]
                print(wristXY)
                if wristPose[0] > 1 or wristPose[0] < 0:
                    return 
                if wristPose[1] > 1 or wristPose[1] < 0:
                    return 
                self.wristPose = self._cam.get_actual_pose(wristXY[0], wristXY[1], self._cam.get_depth_value(wristXY[0], wristXY[1], obj.input_depth_image))
                

            if self._record_start_pose is None:
                self._record_start_pose = self.wristPose
            
            if self.wristPose[0] == 0 and self.wristPose[1] == 0 and self.wristPose[2] == 0:
                return 
            # print(self.wristPose)
            # print(self._record_start_pose)

            diff = [int(self.wristPose[i]) - int(self._record_start_pose[i]) for i in range(3)]
            # print(self.wristPose[2], self._record_start_pose[2])
            # print(int(self.wristPose[2]) - int(self._record_start_pose[2]))


            # if self._last_diff is None:
            #     self._last_diff = diff
            # else:
            # 记录上次的数据
            if self._last_diff is not None:
                _dx = diff[0] - self._last_diff[0]
                _dy = diff[1] - self._last_diff[1]
                _dz = diff[2] - self._last_diff[2]
                _MAX_DIFF = 150
                if abs(_dx) > _MAX_DIFF:
                    diff[0] = self._last_diff[0]
                if abs(_dy) > _MAX_DIFF:
                    diff[1] = self._last_diff[1]
                if abs(_dz) > _MAX_DIFF:
                    diff[2] = self._last_diff[2]
            
            print("Gesture diff", diff[0], diff[1], diff[2])
            self.puber.send_string(f"{diff[0]},{diff[1]},{diff[2]}") 
            self._last_diff = deepcopy(diff)
                

            """
            if self.__t.get(gesture, None) is None:
                # self._db["gesture"] = gesture
                self.__t[gesture] = now
                self._db["dis"] = dis
                self.__t["dis"] = now
                continue
            # db_gesture = self._db.get("gesture")
            t_gesture = self.__t.get("gesture")
            db_dis = self._db.get("dis")
            t_dis = self.__t.get("dis")

            # if gesture == self.FuncNameLists.VICTORY.value:
            #     print("Victory")
            # if gesture == self.FuncNameLists.THUMB_UP.value:
            #     print("Thumb_up")


            # 手势触发
            # if gesture != db_gesture and  gesture in self._callbackFuncDict.keys() :
            if gesture in self._callbackFuncDict.keys() :
                callbacks = self._callbackFuncDict.get(gesture)
                for k, v in enumerate(callbacks):
                    if now - t_gesture >= self.duration.get(gesture)[k]:
                        self._callbackFuncDict.get(gesture)[k]()
                        self._db[gesture] = gesture
                        self.__t[gesture] = None

            
            # 距离检测
            if self.FuncNameLists.INCREASE.value in self._callbackFuncDict.keys() or self.FuncNameLists.REDUCE.value in self._callbackFuncDict.keys():
                diff = dis - db_dis
                # print(round(diff, 4), round(dis, 4))
                if diff > 0:
                    callbacks = self._callbackFuncDict.get(self.FuncNameLists.INCREASE.value)
                    for k, v in enumerate(callbacks):
                        if abs(diff) > self.volatuationData.get(self.FuncNameLists.INCREASE.value)[k]:
                            if (now - t_dis) > self.duration.get(self.FuncNameLists.INCREASE.value)[k]:
                                self._callbackFuncDict.get(self.FuncNameLists.INCREASE.value)[k]()
                                self._db["dis"] = dis
                                self.__t["dis"] = now
                                print("inc", round(diff, 4), round(dis, 4))
                else:
                    callbacks = self._callbackFuncDict.get(self.FuncNameLists.REDUCE.value)
                    for k, v in enumerate(callbacks):
                        if abs(diff) > self.volatuationData.get(self.FuncNameLists.REDUCE.value)[k]:
                            if (now - t_dis) > self.duration.get(self.FuncNameLists.REDUCE.value)[k]:
                                self._callbackFuncDict.get(self.FuncNameLists.REDUCE.value)[k]()
                                self._db["dis"] = dis
                                self.__t["dis"] = now
                                print("dec", round(diff, 4), round(dis, 4))
            """


    def start_record_and_cali(self):
        # 开始
        # print(f"start | {time.time()}")
        if self._cali_flag == False:
            print("Start Record And Cali Gesture")
            self._cali_flag = True
            self._record_start_pose = None

    def stop_record_and_cali(self):
        if self._cali_flag == True:
            print("Stop Record And Cali Gesture")
            self._cali_flag = False
            # 目前该调用无实际作用，只是为了查看数据处理时机
            client.ServerProxy("http://127.0.0.1:9121/", allow_none=True).setData()

                        




        
        
        

        



        