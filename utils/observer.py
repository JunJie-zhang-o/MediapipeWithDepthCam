


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
            if self.name == GestureObserver.FuncNameLists.INCREASE.value:
                # print(self.name, data)
                pass
            if data is not None and v is not None and volatuationData is not None:
                diff = v - data
                if abs(diff) <= volatuationData: 
                    # self._t[k] = None
                    continue                   
                # 绝对值只能判断是否波动,加上差值的判断
                if diffIsPositiveValue:
                    if diff < 0: continue
                if not diffIsPositiveValue:
                    if diff > 0: continue

            
            # 持续时间内,记录首次触发的时间和数据
            dbT = self._t.get(k, None)
            if dbT is None:
                self._t.update({k: t})
                if v is not None:
                    self._data.update({k: v})
                continue
            
            # 时间和数据都满足,判断是否为持续触发和满足时间周期
            if t - dbT >= duration:
                callback()
                if self.name == GestureObserver.FuncNameLists.INCREASE.value:
                    # print(self.name, t, dbT, v, data)
                    pass
                self._t.update({k: t})
                self._data.update({k: v})
                trigger_result.append(True)
        return trigger_result
                

    def reset(self):
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
        CLOSE         = "Closed_Palm"           # 关闭手掌
        POINTING_UP   = "Pointing_Up"           # 食指指天
        THUMB_DOWN    = "Thumb_Down"            # 大拇指向上
        THUMB_UP      = "Thumb_Up"              # 大拇指向下
        VICTORY       = "Victory"               # yeah
        LOVE          = "ILoveYou"              # 1、3、5手指向上,手掌面向或手背面向相机都可以
        INCREASE      = "Increase"              # 大拇指和食指间距拉长
        REDUCE        = "Reduce"                # 大拇指和食指间距减小

    def __init__(self) -> None:
        
        # self.duration = {}
        # self.volatuationData = {}
        self._callbackFuncDict:Dict[str, ActionTrigger] = {}

        self.__t = {}
        self._db = {}

    
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
        t = 0
        while 1:
            time.sleep(0.001)
            now = time.time()
            if type(obj.result) is not GestureRecognizerResultData:
                continue
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
                    continue 
                # db_dis = self._db.get("dis")
                # diff = dis - db_dis
                # if diff > 0:
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
                        




        
        
        

        



        