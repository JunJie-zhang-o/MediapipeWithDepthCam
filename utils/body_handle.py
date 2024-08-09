



import time
from utils.image_list import ImageList
from utils.landmark import PoseLandMarkDetector
from utils.landmark import HandIndex, PoseIndex
from utils.pose import get_user_frame, pose_inv, pose_mul,Pose
from utils.realsense import RealSense
import zmq



class BodyObserver:
    """
        身体数据观察者
    """
    def __init__(self, image_deque:ImageList, cam:RealSense) -> None:
        
        self.image_deque = image_deque
        self._cam = cam
        self._center_frame, self._record_start_pose = None, None
        self.__height, self.__width = None, None
        self._cali_flag = False         
        self.puber = zmq.Context().socket(zmq.PUB)
        self.puber.bind("tcp://*:5556")
        self.puber.set_hwm(100)

    

    def updata(self, obj:PoseLandMarkDetector):

        def checkPixelValid(data):
            # 检测归一化的数据是否超出图像区域, #!当躯干检测时,会推理出某个点在图像外的情况 
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

            if not self._cali_flag:
                return 

            # 获取mediapipe检测的归一化数据,并检测数据有效性
            left_shoulder = obj.result.getKeyPointData(PoseIndex.LEFT_SHOULDER)[0]
            right_shoulder = obj.result.getKeyPointData(PoseIndex.RIGHT_SHOULDER)[0]
            # right_elbow = obj.result.getKeyPointData(PoseIndex.RIGHT_ELBOW)[0]
            nose = obj.result.getKeyPointData(PoseIndex.NOSE)[0]
            right_wrist = obj.result.getKeyPointData(PoseIndex.RIGHT_WRIST)[0]
            if not (checkPixelValid(left_shoulder) and checkPixelValid(right_shoulder) and checkPixelValid(nose) and checkPixelValid(right_wrist)):
                return 
            print('----------------------')
            # 转换为像素坐标
            left_shoulder_xy = [int(left_shoulder["x"]*self.__width), int(left_shoulder["y"]*self.__height)]
            right_shoulder_xy = [int(right_shoulder["x"]*self.__width), int(right_shoulder["y"]*self.__height)]
            # right_elbow_xy = [int(right_elbow["x"]*self.__width), int(right_elbow["y"]*self.__height)]
            nose_xy = [int(nose["x"]*self.__width), int(nose["y"]*self.__height)]
            right_wrist_xy = [int(right_wrist["x"]*self.__width), int(right_wrist["y"]*self.__height)]
            # 计算出实际坐标
            left_shoulder_point = self._cam.get_actual_pose(left_shoulder_xy[0], left_shoulder_xy[1], self._cam.get_depth_value(left_shoulder_xy[0], left_shoulder_xy[1], obj.input_depth_image))
            right_shoulder_point = self._cam.get_actual_pose(right_shoulder_xy[0], right_shoulder_xy[1], self._cam.get_depth_value(right_shoulder_xy[0], right_shoulder_xy[1], obj.input_depth_image))
            # right_elbow_point = self.realsense.get_actual_pose(right_elbow_xy[0], right_elbow_xy[1], self.realsense.get_depth_value(right_elbow_xy[0], right_elbow_xy[1], obj.input_depth_image))
            nose_point = self._cam.get_actual_pose(nose_xy[0], nose_xy[1], self._cam.get_depth_value(nose_xy[0], nose_xy[1], obj.input_depth_image))
            right_wrist_point = self._cam.get_actual_pose(right_wrist_xy[0], right_wrist_xy[1], self._cam.get_depth_value(right_wrist_xy[0], right_wrist_xy[1], obj.input_depth_image))

            #! 偶尔会出现获取数据全为0的情况
            if right_wrist_point[0] == 0:
                print(right_wrist_point)
                return 
            center = [0,0,0]
            center[0] = (right_shoulder_point[0] + left_shoulder_point[0])/2
            center[1] = (right_shoulder_point[1] + left_shoulder_point[1])/2
            center[2] = (right_shoulder_point[2] + left_shoulder_point[2])/2
            # print(center)
            if self._center_frame is None:
                # 以右肩为中心点, 右肩到右肘为X正方向, 右肩到左肩为Y正方向,  
                # self.center_frame = get_user_frame(right_shoulder_point, right_elbow_point, left_shoulder_point)
                # 以左右肩为中心点,中心点到左肩为X正方向,中心点到鼻子为Y正方向
                self._center_frame = get_user_frame(center, left_shoulder_point, nose_point)
                # 修正坐标系与相机坐标系进行对齐
                self._center_frame.rx = -3.14
                self._center_frame.ry = 0
                self._center_frame.rz = 0
            # 获取右手手腕在中心坐标系的位置
            wrist_in_cf_pose = self.get_wrist_in_center_frame(right_wrist_point)
            if self._record_start_pose is None:
                self._record_start_pose = wrist_in_cf_pose
            
            diff = [wrist_in_cf_pose[i] - self._record_start_pose[i] for i in range(3)]

            print("diff", diff[0], diff[1], diff[2])
            self.puber.send_string(f"{diff[0]},{diff[1]},{diff[2]}")    
            # exit()



    def get_center_frame(self):
        return self._center_frame

    
    def start_record_and_cali(self):
        # 开始
        print("Start Record And Cali")
        self._center_frame = None
        self._cali_flag = True
        self._record_start_pose = None


    def stop_record_and_cali(self):
        print("Stop Record And Cali")
        self._cali_flag = False



    def get_wrist_in_center_frame(self, wrist_point):
        diff = pose_mul(pose_inv(self._center_frame), [wrist_point[0],wrist_point[1],wrist_point[2],0,0,0])
        return diff
