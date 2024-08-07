#!/usr/bin/env python3
# coding=utf-8
'''
Author       : zhangjunjie jay.zhangjunjie@outlook.com
Date         : 2024-08-07 22:20:06
LastEditTime : 2024-08-08 00:39:25
LastEditors  : zhangjunjie jay.zhangjunjie@outlook.com
Description  : 
'''

import cv2
import numpy as np
from pyorbbecsdk import Config, OBAlignMode, OBFormat, OBSensorType, Pipeline, get_version, get_stage_version
from pyorbbecsdk import Context,OBLogLevel
import atexit


class Gemini2:
    Context.set_logger_to_console(OBLogLevel.NONE)

    def __init__(self,  enable_depth_image=False) -> None:
        
        self.config = Config()
        self.pipeline = Pipeline()


        color_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        depth_profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)

        # color_profile = color_profile_list.get_video_stream_profile(img_size[0], img_size[1], OBFormat.RGB, framerate)
        # depth_profile = depth_profile_list.get_video_stream_profile(img_size[0], img_size[1], OBFormat.Y16, framerate)
        color_profile = color_profile_list.get_video_stream_profile(640, 480, OBFormat.RGB, 30)
        depth_profile = depth_profile_list.get_video_stream_profile(640, 400, OBFormat.Y16, 30)

        self.config.enable_stream(color_profile)
        self.config.enable_stream(depth_profile)


        self.start_pipeline_flag = False

        # profile_list = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        # # depth_profile = profile_list.get
        # print(profile_list)
        # self._find_device()
        self.depth_scale = None     # 深度比例
        self.intrinsics = None      # 相机内参
        self.align_mode = None      # 对齐方式
        pass


    def start_pipeline(self):
        self.pipeline.start(self.config)
        self.start_pipeline_flag = True

        atexit.register(lambda: self.pipeline.stop())

        self.intrinsics = self.pipeline.get_camera_param()
        # self.depth_scale = self.pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()  # 获取深度比例,用于深度距离转换到真实物理空间距离


    def get_frame(self,  is_get_depth_frame=True, is_get_color_frame=True, is_convert_np_array=True):
        """
            获取数据帧
        """
        depth_frame, color_frame = None, None
        if not self.start_pipeline_flag:
            self.start_pipeline()
        frames = self.pipeline.wait_for_frames()
        if is_get_color_frame:
            color_frame = frames.get_color_frame()
        if is_get_depth_frame:
            depth_frame = frames.get_depth_frame()
            if self.depth_scale is None: self.depth_scale = depth_frame.get_depth_scale()
        if is_convert_np_array:
            ret_depth_array, ret_color_array = None, None
            if depth_frame is not None:
                ret_depth_array = np.asanyarray(depth_frame.get_data())
            if color_frame is not None:
                ret_color_array = np.asanyarray(color_frame.get_data())
            return ret_depth_array, ret_color_array
        else:
            return depth_frame, color_frame
    

    def get_align_frame(self, is_get_depth_frame=True, is_get_color_frame=True, is_convert_np_array=True):
        """
            获取对齐后的数据帧
        """
        depth_frame, color_frame = None, None
        if not self.start_pipeline_flag:
            self.start_pipeline()
        while 1:
            frames = self.pipeline.wait_for_frames(100)
            if frames is not None:
                if frames.get_color_frame() is None:
                    # print("")
                    continue
                break
        if is_get_color_frame:
            color_frame = frames.get_color_frame()
        if is_get_depth_frame:
            depth_frame = frames.get_depth_frame()
            if self.depth_scale is None: self.depth_scale = depth_frame.get_depth_scale()
        if is_convert_np_array:
            ret_depth_array, ret_color_array = None, None
            if depth_frame is not None:
                depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                depth_data = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))
                depth_data = depth_data.astype(np.float32) * depth_frame.get_depth_scale()
                ret_depth_array = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            if color_frame is not None:
                data = np.asanyarray(color_frame.get_data())
                image = np.resize(data, (color_frame.get_height(), color_frame.get_width(), 3))
                ret_color_array = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
            return ret_depth_array, ret_color_array
        else:
            return depth_frame, color_frame


    def set_align_mode(self, align_mode=OBAlignMode.HW_MODE):
        """设置对齐模式

        Args:
            align_mode (_type_, optional): _description_. Defaults to rs.stream.color.
        """
        # self.align_mode = rs.align(align_mode)
        self.config.set_align_mode(align_mode)
        self.pipeline.enable_frame_sync()

    
    def get_depth_value(self, x, y, depth_frame):
        """从深度图中获取指定位置的深度信息
        """
        pass


    def get_actual_pose(self, x, y, z):
        # 根据图像上的x,y以及实际的深度信息,结合相机内参 计算出真实的物理值
        pass
    
    
    def _find_device(self):
        context = Context()
        device_list = context.query_devices()
        print("Found Orbbec Camera:")
        dev_count = device_list.get_count()
        if dev_count == 0:
            print("No device connected")
            return
        else:
            # dev_count = dev
            for i in range(dev_count):
                dev = device_list.get_device_by_index(i)
                print(" Device info: {}".format(dev.get_device_info()))
                print(" Sensor list:")
                sensor_list = dev.get_sensor_list()
                for i in range(sensor_list.get_count()):
                    sensor = sensor_list.get_sensor_by_index(i)
                    print("     {}".format(sensor.get_type()))




if __name__ == "__main__":

    cam = Gemini2()