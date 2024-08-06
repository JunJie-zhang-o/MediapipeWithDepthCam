



import time
from utils.image_list import ImageList
from utils.landmark import PoseLandMarkDetector
from utils.landmark import HandIndex, PoseIndex
from utils.pose import get_user_frame, pose_inv, pose_mul,Pose
from utils.realsense import RealSense
import zmq



class BodyObserver:


    def __init__(self, image_deque:ImageList,realsense:RealSense) -> None:
        
        self.image_deque = image_deque
        self.realsense = realsense
        self.center_frame = None
        self.__height, self.__width = None, None
        self.puber = zmq.Context().socket(zmq.PUB)
        self.puber.bind("tcp://*:5556")
        self.puber.set_hwm(100)

    

    def updata(self, obj:PoseLandMarkDetector):

        def checkPixelValid(data):
            if data["x"] > 1 or data["x"] < 0:
                return False
            if data["y"] > 1 or data["y"] < 0:
                return False
            return True
        timestamp = 0
        # while 1:
            # time.sleep(0.001)
        if obj.result is not None:
            if timestamp != obj.input_timestamp:
                timestamp = obj.input_timestamp
            else:
                return 
            if self.__height is None:
                self.__height, self.__width = obj.output_image.shape[:2]

            # 获取mediapipe检测的归一化数据
            left_shoulder = obj.result.getKeyPointData(PoseIndex.LEFT_SHOULDER)[0]
            right_shoulder = obj.result.getKeyPointData(PoseIndex.RIGHT_SHOULDER)[0]
            # right_elbow = obj.result.getKeyPointData(PoseIndex.RIGHT_ELBOW)[0]
            nose = obj.result.getKeyPointData(PoseIndex.NOSE)[0]
            right_wrist = obj.result.getKeyPointData(PoseIndex.RIGHT_WRIST)[0]
            if not (checkPixelValid(left_shoulder) and checkPixelValid(right_shoulder) and checkPixelValid(nose) and checkPixelValid(right_wrist)):
                # pass
            # else:
                return 
            print('----------------------')
            # 转换为像素坐标
            left_shoulder_xy = [int(left_shoulder["x"]*self.__width), int(left_shoulder["y"]*self.__height)]
            right_shoulder_xy = [int(right_shoulder["x"]*self.__width), int(right_shoulder["y"]*self.__height)]
            # right_elbow_xy = [int(right_elbow["x"]*self.__width), int(right_elbow["y"]*self.__height)]
            nose_xy = [int(nose["x"]*self.__width), int(nose["y"]*self.__height)]
            right_wrist_xy = [int(right_wrist["x"]*self.__width), int(right_wrist["y"]*self.__height)]
            # 计算出实际坐标
            left_shoulder_point = self.realsense.get_actual_pose(left_shoulder_xy[0], left_shoulder_xy[1], self.realsense.get_depth_value(left_shoulder_xy[0], left_shoulder_xy[1], obj.input_depth_image))
            right_shoulder_point = self.realsense.get_actual_pose(right_shoulder_xy[0], right_shoulder_xy[1], self.realsense.get_depth_value(right_shoulder_xy[0], right_shoulder_xy[1], obj.input_depth_image))
            # right_elbow_point = self.realsense.get_actual_pose(right_elbow_xy[0], right_elbow_xy[1], self.realsense.get_depth_value(right_elbow_xy[0], right_elbow_xy[1], obj.input_depth_image))
            nose_point = self.realsense.get_actual_pose(nose_xy[0], nose_xy[1], self.realsense.get_depth_value(nose_xy[0], nose_xy[1], obj.input_depth_image))
            right_wrist_point = self.realsense.get_actual_pose(right_wrist_xy[0], right_wrist_xy[1], self.realsense.get_depth_value(right_wrist_xy[0], right_wrist_xy[1], obj.input_depth_image))

            center = [0,0,0]
            center[0] = (right_shoulder_point[0] + left_shoulder_point[0])/2
            center[1] = (right_shoulder_point[1] + left_shoulder_point[1])/2
            center[2] = (right_shoulder_point[2] + left_shoulder_point[2])/2
            # print(center)
            if self.center_frame is None:
                # self.center_frame = get_user_frame(right_shoulder_point, right_elbow_point, left_shoulder_point)
                self.center_frame = get_user_frame(center, left_shoulder_point, nose_point)
                self.center_frame.rx = -3.14
                self.center_frame.ry = 0
                self.center_frame.rz = 0
            # print(self.center_frame)
            diff = self.get_wrist_in_center_frame(right_wrist_point)
            # print(right_wrist_xy)
            print(right_wrist_point)
            if right_wrist_point[0] == 0:
                # print("-")
                # continue
                return 
            print(diff[0], diff[1], diff[2])
            self.puber.send_string(f"{diff[0]},{diff[1]},{diff[2]}")
            # exit()



    def get_center_frame(self):
        pass


    def get_wrist_in_center_frame(self, wrist_point):
        diff = pose_mul(pose_inv(self.center_frame), [wrist_point[0],wrist_point[1],wrist_point[2],0,0,0])
        return diff
