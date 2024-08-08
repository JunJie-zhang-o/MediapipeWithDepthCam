


from abc import ABC, abstractmethod
import math
from threading import Thread
import time
from typing import Dict
from enum import Enum

from utils.landmark import GestureLandMarkDetector, GestureRecognizerResultData


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




class GestureObserver(Observer):
    """
        手势特征观察者
    """
    class FuncNameLists(Enum):
        UNKONWN       = "UnKonwn"               # 未知手势
        OPEN          = "Open_Palm"             # 打开手掌
        CLOSE         = "Closed_Palm"           # 关闭手掌
        POINTING_UP   = "Pointing_Up"           # 食指指天
        THUMB_DOWN    = "Thumb_Down"            # 大拇指向上
        THUMB_UP      = "Thumb_Up"              # 大拇指向下
        VICTORY       = "Victory"               # yeah
        LOVE          = "ILoveYou"              # 1、3、5手指向上,手掌面向或手背面向相机都可以
        INCREASE      = "Increase"              # 大拇指和食指间距拉长
        REDUCE        = "Reduce"                # 大拇指和食指间距减小

    def __init__(self) -> None:
        
        self.duration = {}
        self.volatuationData = {}
        self._callbackFuncDict:dict[str:list] = {}

        self.__t = {}
        self._db = {}

    
    def register_callback(self, name:FuncNameLists, callback, duration, volatuationData):
        if name in self.FuncNameLists:    
            if name in self._callbackFuncDict.keys():
                self._callbackFuncDict[name].append(callback)
                self.duration[name].append(duration)
                self.volatuationData[name].append(volatuationData)
            else:
                self._callbackFuncDict[name] = [callback]
                self.duration[name] = [duration]
                self.volatuationData[name] = [volatuationData]
        else:
            print("name not in funcNameLists")
            exit(0)


    def deregister_callback(self, name):
        self._callbackFuncDict.pop(name)

    
    def updata(self, obj:GestureLandMarkDetector):

        while 1:
            time.sleep(0.001)
            now = time.time()
            if type(obj.result) is not GestureRecognizerResultData:
                continue
            gesture = obj.result.gestures.category_name
            dis = obj.get_thumb_indexfinger_tip_dis()
            if not self._db:
                self._db["gesture"] = gesture
                self.__t["gesture"] = now
                self._db["dis"] = dis
                self.__t["dis"] = now
                continue
            db_gesture = self._db.get("gesture")
            t_gesture = self.__t.get("gesture")
            db_dis = self._db.get("dis")
            t_dis = self.__t.get("dis")

            # 手势触发
            if gesture != db_gesture and  gesture in self._callbackFuncDict.keys() :
                callbacks = self._callbackFuncDict.get(gesture)
                for k, v in enumerate(callbacks):
                    if gesture != db_gesture and now - t_gesture >= self.duration.get("gesture")[k]:
                        self._callbackFuncDict.get("gesture")[k]()
                        self._db["gesture"] = gesture
                        self.__t["gesture"] = now

            
            # 距离检测
            if self.FuncNameLists.INCREASE in self._callbackFuncDict.keys() or self.FuncNameLists.REDUCE in self._callbackFuncDict.keys():
                diff = dis - db_dis
                # print(round(diff, 4), round(dis, 4))
                if diff > 0:
                    callbacks = self._callbackFuncDict.get(self.FuncNameLists.INCREASE)
                    for k, v in enumerate(callbacks):
                        if abs(diff) > self.volatuationData.get(self.FuncNameLists.INCREASE)[k]:
                            if (now - t_dis) > self.duration.get(self.FuncNameLists.INCREASE)[k]:
                                self._callbackFuncDict.get(self.FuncNameLists.INCREASE)[k]()
                                self._db["dis"] = dis
                                self.__t["dis"] = now
                                print("inc", round(diff, 4), round(dis, 4))
                else:
                    callbacks = self._callbackFuncDict.get(self.FuncNameLists.REDUCE)
                    for k, v in enumerate(callbacks):
                        if abs(diff) > self.volatuationData.get(self.FuncNameLists.REDUCE)[k]:
                            if (now - t_dis) > self.duration.get(self.FuncNameLists.REDUCE)[k]:
                                self._callbackFuncDict.get(self.FuncNameLists.REDUCE)[k]()
                                self._db["dis"] = dis
                                self.__t["dis"] = now
                                print("dec", round(diff, 4), round(dis, 4))
                    
                        




        
        
        

        



        