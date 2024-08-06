


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyrealsense2 as rs
# from utils.fps import FPS


class RealSense:


    def __init__(self, img_size=(640, 480), framerate=90, enable_depth_image=False) -> None:
        self.selected_devices = []              # Store connected device(s)
        self._find_devices()
        self._find_depth_and_rgb_sensors()

        self.pipeline = rs.pipeline()
        self.cfg = rs.config()
        # self.profile = self.pipe.start(self.cfg)

        self.cfg.enable_stream(rs.stream.depth, img_size[0], img_size[1], rs.format.z16, framerate)
        self.cfg.enable_stream(rs.stream.color, img_size[0], img_size[1], rs.format.bgr8, framerate)

        self.start_pipeline_flag = False

        self.enable_depth_frame = enable_depth_image
        if not self.enable_depth_frame:
            # self.cfg.disable_stream(rs.RS2_STREAM_DEPTH)
            pass

        
        self.depth_scale = None     # 深度比例
        self.intrinsics = None      # 相机内参
        self.align_mode = None      # 对齐方式


    def start_pipeline(self):
        self.pipeline.start(self.cfg)
        self.start_pipeline_flag = True

        self.depth_scale = self.pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()  # 获取深度比例,用于深度距离转换到真实物理空间距离
        self.intrinsics = self.pipeline.get_active_profile().get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    

    def get_frame(self, is_get_depth_frame=True, is_get_color_frame=True, is_convert_np_array=True):
        depth_frame, color_frame = None, None
        if not self.start_pipeline_flag:
            self.start_pipeline()
        frames = self.pipeline.wait_for_frames()
        # if self.enable_depth_frame:
            # depth_frame = frames.get_depth_frame()
        if is_get_color_frame:
            color_frame = frames.get_color_frame()
        if is_get_depth_frame:
            depth_frame = frames.get_depth_frame()
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
        if self.align_mode:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align_mode.process(frames)
        else:
            raise Exception("Must Set Align Mode | May Be rs.stream.color or rs.stream.depth")
        depth_frame, color_frame = None, None
        if is_get_color_frame:
            color_frame = aligned_frames.get_color_frame()
        if is_get_depth_frame:
            depth_frame = aligned_frames.get_depth_frame()
        if is_convert_np_array:
            ret_depth_array, ret_color_array = None, None
            if depth_frame is not None:
                ret_depth_array = np.asanyarray(depth_frame.get_data())
            if color_frame is not None:
                ret_color_array = np.asanyarray(color_frame.get_data())
            return ret_depth_array, ret_color_array
        else:
            return depth_frame, color_frame
    

    def set_align_mode(self, align_mode=rs.stream.color):
        self.align_mode = rs.align(align_mode)


    def get_depth_value(self, x, y, depth_frame):
        return depth_frame[y,x] * self.depth_scale


    def get_coordinate_from():
        # 根据图像上的x,y以及实际的深度信息,结合相机内参 计算出真实的物理值
        pass

    def _find_devices(self):
        print("Searching Devices..")
        for d in rs.context().devices:
            self.selected_devices.append(d)
            print(d.get_info(rs.camera_info.name))
        if not self.selected_devices:
            print("No RealSense device is connected!")


    def _find_depth_and_rgb_sensors(self):
        rgb_sensor = depth_sensor = None

        for device in self.selected_devices:                         
            print("Required sensors for device:", device.get_info(rs.camera_info.name))
            for s in device.sensors:                              # Show available sensors in each device
                if s.get_info(rs.camera_info.name) == 'RGB Camera':
                    print(" - RGB sensor found")
                    rgb_sensor = s                                # Set RGB sensor
                if s.get_info(rs.camera_info.name) == 'Stereo Module':
                    depth_sensor = s                              # Set Depth sensor
                    print(" - Depth sensor found")



if __name__ == "__main__":


    realSense = RealSense()
    # realSense.cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 90)
    # realSense.cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 90)
    # fps = FPS()
    try:
        while True:
            # fps.refresh()
            depth_frame, color_frame = realSense.get_frame()
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape

            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                images = np.hstack((resized_color_image, depth_colormap))
            else:
                images = np.hstack((color_image, depth_colormap))

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            # print(fps.fps)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

    finally:

        # Stop streaming
        realSense.pipeline.stop()