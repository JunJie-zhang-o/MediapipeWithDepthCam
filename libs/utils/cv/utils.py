
import os
import platform
from queue import Queue
import re
from threading import Thread
import time
import cv2




def print_cv_build_info():
    """打印Opencv的构建信息
    """
    print(cv2.getBuildInformation())  


def isSupportGStreamer():
    """查询opencv是否支持GStreamer
    """
    info = cv2.getBuildInformation().split("\n")
    for i in info:
        if i.find("GStreamer") >= 0:
            return True if "True" in i else False




def print_cam_base_info(cap: cv2.VideoCapture):
    """
    打印相机的基本参数

    Args:
        cap (cv2.VideoCapture): cv2.VideoCapture instance
    """
    def fourcc_to_string(fourcc):
        # 解析四字符码
        return "".join([chr((int(fourcc) >> 8 * i) & 0xFF) for i in range(4)])

    width                   = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height                  = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps                     = cap.get(cv2.CAP_PROP_FPS)
    formats                 = fourcc_to_string(cap.get(cv2.CAP_PROP_FOURCC))

    brightness              = cap.get(cv2.CAP_PROP_BRIGHTNESS)                          # 亮度 min=-64 max=64 step=1 default=0 value=0
    contrast                = cap.get(cv2.CAP_PROP_CONTRAST)                            # 对比度 min=0 max=100 step=1 default=40 value=40
    saturation              = cap.get(cv2.CAP_PROP_SATURATION)                          # 饱和度 min=0 max=100 step=1 default=64 value=64
    hue                     = cap.get(cv2.CAP_PROP_HUE)                                 # 色调 min=-180 max=180 step=1 default=0 value=0
    gain                    = cap.get(cv2.CAP_PROP_GAIN)                                # 增益 min=1 max=128 step=1 default=64 value=64
    exposure                = cap.get(cv2.CAP_PROP_EXPOSURE)                            # 曝光 min=50 max=10000 step=1 default=166 value=166 flags=inactive
    wb                      = cap.get(cv2.CAP_PROP_WB_TEMPERATURE)                      # 白平衡温度 min=2800 max=6500 step=10 default=4600 value=4300
    autoWB                  = cap.get(cv2.CAP_PROP_AUTO_WB)                             # 自动白平衡 default=1 value=0
    sharpness               = cap.get(cv2.CAP_PROP_SHARPNESS)                           # 锐度 min=0 max=100 step=1 default=50 value=50
    gamma                   = cap.get(cv2.CAP_PROP_GAMMA)                               # 伽马值 min=100 max=500 step=1 default=300 value=300
    # powerFramrate           = cap.get(cv2.CAP_PROP_XI_PRM_ACQ_FRAMERATE)                       # 电源线频率# 0 = 50 Hz, 1 = 60 Hz, 2 = 禁用 min=0 max=2 default=1 value=1
    backLight               = cap.get(cv2.CAP_PROP_BACKLIGHT)                           # 背光补偿 min=0 max=2 step=1 default=0 value=0
    autoExposure            = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)                       # 自动曝光 min=0 max=3 default=3 value=3
    # exposureAutoPriority    = cap.get(cv2.CAP_PROP_EXPOSURE_AUTO_PRIORITY)                     # 自动曝光优先级（0 = 关闭, 1 = 开启）
    focus                   = cap.get(cv2.CAP_PROP_FOCUS)                               # 绝对焦点值（min=0 max=1023 step=1 default=68 value=68 flags=inactive
    autoFocus               = cap.get(cv2.CAP_PROP_AUTOFOCUS)                           # 自动对焦 default=1 value=1

    # backendName = cap.getBackendName()
    exceptionMode = cap.getExceptionMode()
    print("------------------------------------------")
    print(
          "Cam 宽度      :"+f"{width} \n"
        + "Cam 高度      :"+f"{height} \n"
        + "Cam FPS       :"+f"{fps} \n"
        + "Cam 格式      :"+f"{formats} \n"
        # + "Cam 后端    : {backendName} \n"
        + "Cam 异常模式  :"+f"{exceptionMode} \n"
        + "Cam 亮度      :"+f"{brightness} \n"
        + "Cam 对比度    :"+f"{contrast} \n" 
        + "Cam 饱和度    :"+f"{saturation} \n"
        + "Cam 色调      :"+f"{hue} \n"
        + "Cam 自动白平衡:"+f"{autoWB} \n"
        + "Cam 伽马      :"+f"{gamma} \n"
        + "Cam 增益      :"+f"{gain} \n"
        + "Cam 白平衡温度:"+f"{wb} \n"
        + "Cam 锐度      :"+f"{sharpness} \n"
        + "Cam 背光补偿  :"+f"{backLight} \n"
        + "Cam 自动曝光  :"+f"{autoExposure} \n"
        + "Cam 曝光      :"+f"{exposure} \n"
        # + "Cam 电源频率     :{powerFramrate} \n"
        # + "Cam 自动曝光优先级:{exposureAutoPriority} \n"
        + "Cam 焦距      :"+f"{focus} \n"
        + "Cam 自动焦距  :"+f"{autoFocus} \n"
    )
    print("------------------------------------------")


def set_cam_base_params(
    cap: cv2.VideoCapture,
    width: int = None,
    height: int = None,
    fps: int = None,
    formats: int = None,
    whiteBalance:int = None,
    autoWhiteBalance:int=None,
    brightness:int = None,
    contrast:int =None,
    saturation:int = None,
    hue:int = None,
    gain:int = None ,
    exposure:int=None,
    sharpness:int = None,
    gamma:int=None,
    backLight:int=None,
    autoExposure:int=None,
    focus:int=None,
    autoFocus:int=None,
):

    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps is not None:
        cap.set(cv2.CAP_PROP_FPS, fps)
    if formats is not None:
        fourcc = cv2.VideoWriter.fourcc(*formats)
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    if brightness is not None:       # 亮度 min=-64 max=64 step=1 default=0 value=0
        cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
    if contrast is not None:         # 对比度 min=0 max=100 step=1 default=40 value=40
        cap.set(cv2.CAP_PROP_CONTRAST, contrast)
    if saturation is not None:       # 饱和度 min=0 max=100 step=1 default=64 value=64
        cap.set(cv2.CAP_PROP_SATURATION, saturation)
    if hue is not None:              # 色调 min=-180 max=180 step=1 default=0 value=0
        cap.set(cv2.CAP_PROP_HUE, hue)
    if autoWhiteBalance is not None: # 自动白平衡 default=1 value=0
        cap.set(cv2.CAP_PROP_AUTO_WB, autoWhiteBalance)
    if gamma is not None:            # 伽马值 min=100 max=500 step=1 default=300 value=300
        cap.set(cv2.CAP_PROP_GAMMA, gamma) 
    if gain is not None:             # 增益 min=1 max=128 step=1 default=64 value=64
        cap.set(cv2.CAP_PROP_GAIN, gain)
    if whiteBalance is not None:     # 白平衡温度 min=2800 max=6500 step=10 default=4600 value=4300
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, whiteBalance)
    if sharpness is not None:        # 锐度 min=0 max=100 step=1 default=50 value=50
        cap.set(cv2.CAP_PROP_SHARPNESS, sharpness)
    if backLight is not None:        # 背光补偿 min=0 max=2 step=1 default=0 value=0
        cap.set(cv2.CAP_PROP_BACKLIGHT, backLight)      
    if autoExposure is not None:     # 自动曝光 min=0 max=3 default=3 value=3
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, autoExposure)   
    if exposure is not None:         # 曝光 min=50 max=10000 step=1 default=166 value=166 flags=inactive
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    if autoFocus is not None:        # 自动对焦 default=1 value=1
        cap.set(cv2.CAP_PROP_AUTOFOCUS, autoFocus)
    if focus is not None:            # 绝对焦点值（min=0 max=1023 step=1 default=68 value=68 flags=inactive
        cap.set(cv2.CAP_PROP_FOCUS, focus)




class VideoWriter:
    """
        多线程生成视频.

        ! 当分辨率较大时,会导致写入过慢从而导致内存增加到溢出.可以尝试将图像数据暂时保存到文件,最后加载并生成视频. (可以尝试h5py和numpy序列化数据)

    """

    def __init__(self, savePath:str, fps:int, width:int , height:int, fourcc:str = "mp4v",) -> None:
        self._fps = fps
        self._width = width
        self._height = height
        
        self._writer = cv2.VideoWriter(savePath, cv2.VideoWriter_fourcc(*fourcc), self._fps, (self._width, self._height)) 

        self._q = Queue()
        self.writeFromQueueLoop = None


    def write(self, frame):
        width, height = frame.shape[1], frame.shape[0]
        if width == self._width and height == self._height:
            self._writer.write(frame)
        else:
            raise Exception(f"The Write Frame Size is incorrect. | Frame Width:{width} Frame Height:{height} Video Width:{self._width} Video Height:{self._height}")
            exit()


    def putToQueue(self, value):
        self._q.put(value)


    def release(self):
        if self.writeFromQueueLoop is not None:
            while True:
                if(self._q.qsize() == 0):
                    self.writeFromQueueLoop = False
                    break
                time.sleep(0.5)
        self._writer.release()


    def enableWriteFromQueueThread(self):
        self.writeFromQueueLoop = True
        Thread(target=self.__writeFromQueue, daemon=True, name="VideoWriter From Queue").start()


    def __writeFromQueue(self):
        while self.writeFromQueueLoop:
            print(self._q.qsize())
            frame = self._q.get()
            self.write(frame)


    def pack(self):
        pass


    def unpack(self):
        pass




def get_camera_id(camera_name):
    cam_num = None
    if os.name == 'nt':
        cam_num = find_cameras_windows(camera_name)
    elif platform.system() == "Darwin":
        import usb.core
        if "DYLD_LIBRARY_PATH" not in os.environ:
            os.environ['DYLD_LIBRARY_PATH'] = '/opt/homebrew/lib:'
        devices = usb.core.find(find_all=True)
        for idx, device in enumerate(devices):
            if camera_name in device.product:
                cam_num = idx
                break
    else:
        for file in os.listdir("/sys/class/video4linux"):
            real_file = os.path.realpath("/sys/class/video4linux/" + file + "/name")
            with open(real_file, "rt") as name_file:
                name = name_file.read().rstrip()
            if camera_name in name:
                cam_num = int(re.search("\d+$", file).group(0))
                found = "FOUND!"
            else:
                found = "      "
            print("{} {} -> {}".format(found, file, name))
    if cam_num is None:
        print("ERROR! Can't Found Camera Device")
        exit()
    return cam_num

if os.name == 'nt':
    def find_cameras_windows(camera_name):

        from pygrabber.dshow_graph import FilterGraph
        graph = FilterGraph()

        # get the device name
        allcams = graph.get_input_devices() # list of camera device
        description = ""
        for cam in allcams:
            if camera_name in cam:
                description = cam
        try:
            device = graph.get_input_devices().index(description)
        except ValueError as e:
            print("Device is not in this list")
            print(graph.get_input_devices())
            import sys
            sys.exit()

        return device





IsMac = True if platform.system() == "Darwin" else False
IsWindows = True if platform.system() == "Windows" else False
IsLinux = True if platform.system() == "Linux" else False