from dataclasses import dataclass
from enum import Enum
import subprocess
import cv2

import platform

if platform.system() != "Linux":
    print("该脚本仅支持Linux运行!!!")
    exit()


class V4l2ToCv2(Enum):
    BRIGHTNESS = cv2.CAP_PROP_BRIGHTNESS
    CONTRAST = cv2.CAP_PROP_CONTRAST
    SATURATION = cv2.CAP_PROP_SATURATION
    HUE = cv2.CAP_PROP_HUE
    AUTO_WHITE_BALANCE = cv2.CAP_PROP_AUTO_WB
    DO_WHITE_BALANCE = None
    RED_BALANCE = cv2.CAP_PROP_WHITE_BALANCE_RED_V
    BLUE_BALANCE = cv2.CAP_PROP_WHITE_BALANCE_BLUE_U
    GAMMA = cv2.CAP_PROP_GAMMA
    WHITENESS = None
    EXPOSURE = cv2.CAP_PROP_EXPOSURE
    AUTOGAIN = None
    GAIN = cv2.CAP_PROP_GAIN
    WHITE_BALANCE_TEMPERATURE = cv2.CAP_PROP_WB_TEMPERATURE
    WHITE_BALANCE_TEMPERATURE_AUTO = cv2.CAP_PROP_AUTO_WB
    SHARPNESS = cv2.CAP_PROP_SHARPNESS
    BACKLIGHT_COMPENSATION = cv2.CAP_PROP_BACKLIGHT
    AUTOBRIGHTNESS = None
    EXPOSURE_AUTO = cv2.CAP_PROP_AUTO_EXPOSURE
    EXPOSURE_ABSOLUTE = cv2.CAP_PROP_EXPOSURE
    FOCUS_ABSOLUTE = cv2.CAP_PROP_FOCUS
    FOCUS_AUTO = cv2.CAP_PROP_AUTOFOCUS


@dataclass
class IntParam:
    name: str
    min: int
    max: int
    step: int
    default: int
    value: int
    flags: str = None

    @classmethod
    def parse(cls, s: str):

        def parseParams(ps: str):
            params = {}
            for p in ps.split(" "):
                if p != "":
                    k, v = p.split("=")
                    # if v.isdigit():
                    try:
                        params.update({k: int(v)})
                    except:
                        params.update({k: (v)})
            return params

        name = s[: s.find(" ")]
        return name, cls(name, **parseParams(s.split(":")[-1]))

    @classmethod
    def isIntParam(cls, s: str):
        if "(int)" in s:
            return True
        return False


@dataclass
class BoolParam:
    name: str
    min: int
    max: int
    default: bool
    value: bool

    @classmethod
    def parse(cls, s):
        params = {}

        def parseParams(ps: str):
            for p in ps.split(" "):
                if p != "":
                    k, v = p.split("=")
                    params.update({k: bool(v)})
            params.update({"min": 0})
            params.update({"max": 1})
            return params

        name = s[: s.find(" ")]
        return name, cls(name, **parseParams(s.split(":")[-1]))

    @classmethod
    def isBoolParam(cls, s):
        if "(bool)" in s:
            return True
        return False


@dataclass
class MenuParam:
    name: str
    min: int
    max: int
    default: int
    value: int

    @classmethod
    def parse(cls, s):
        params = {}

        def parseParams(ps: str):
            for p in ps.split(" "):
                if p != "":
                    k, v = p.split("=")
                    params.update({k: int(v)})
            return params

        name = s[: s.find(" ")]
        return name, cls(name, **parseParams(s.split(":")[-1]))

    @classmethod
    def isMenuParam(cls, s):
        if "(menu)" in s:
            return True
        return False


class CameraParams:
    WINDOWS_NAME = "Camera Settings"

    def __init__(self, camIndex) -> None:

        self.camIndex = camIndex
        self.params = {}
        self.__getCtrlsListInfo()
        cv2.namedWindow(self.WINDOWS_NAME, cv2.WINDOW_AUTOSIZE)
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.createTrackbar()

    def __getCtrlsListInfo(self):
        result = subprocess.run(
            ["v4l2-ctl", f"--device=/dev/video{self.camIndex}", "--list-ctrls"],
            stdout=subprocess.PIPE,
            text=True,
        )
        print(result.stdout)
        lines = result.stdout.split("\n")

        for line in lines:
            # print(line)
            line = line.lstrip()
            if IntParam.isIntParam(line):
                k, v = IntParam.parse(line)
                self.params.update({k: v})

            elif BoolParam.isBoolParam(line):
                k, v = BoolParam.parse(line)
                self.params.update({k: v})
            elif MenuParam.isMenuParam(line):
                k, v = MenuParam.parse(line)
                self.params.update({k: v})

    def createTrackbar(self):
        for k in self.params.keys():
            # if type(params[k]) == IntParam :
            cv2.createTrackbar(
                k,
                self.WINDOWS_NAME,
                self.params[k].value,
                self.params[k].max,
                self.updateParam,
            )

    def updateParam(self, arg):
        for k in self.params.keys():
            k: str
            v = cv2.getTrackbarPos(k, self.WINDOWS_NAME)
            if v != self.params[k].value:
                if (
                    k.upper() in V4l2ToCv2.__members__.keys()
                    and V4l2ToCv2[k.upper()].value is not None
                ):
                    self.params[k].value = v
                    print(f"设置相机参数 | {k}:{v} | default:{self.params[k].default}")
                    self.cap.set(V4l2ToCv2[k.upper()].value, v)
                else:
                    print(
                        f"未能正确找到设置该参数的接口 | {k}:{v} | default:{self.params[k].default}"
                    )

    def show(self):
        while True:
            # 读取相机帧
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # 显示帧
            cv2.imshow("Camera", frame)

            # 按下q键退出循环
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        # 释放资源
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":

    a = CameraParams(0)
    a.show()
