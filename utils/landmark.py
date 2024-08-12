

from dataclasses import dataclass , asdict
from enum import IntEnum
import math
import time
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import cv2
import numpy as np
from utils.fps import FPS


## -----------------------------------------------------------------------------
# mediapipe 相关
_BaseOptions = mp.tasks.BaseOptions
_VisionRunningMode = mp.tasks.vision.RunningMode

_PoseLandmarker = mp.tasks.vision.PoseLandmarker
_PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
_PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult

_HandLandmarker = mp.tasks.vision.HandLandmarker
_HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
_HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult

_GestureRecognizer = mp.tasks.vision.GestureRecognizer
_GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
_GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult

## -----------------------------------------------------------------------------

def draw_pose_landmarks_on_image(rgb_image, detection_result):
    """
        绘制姿势特征点
    """
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

def draw_hand_landmarks_on_image(rgb_image, detection_result):
    """
        绘制手部特征点
    """
    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
        annotated_image,
        hand_landmarks_proto,
        solutions.hands.HAND_CONNECTIONS,
        solutions.drawing_styles.get_default_hand_landmarks_style(),
        solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image


## -----------------------------------------------------------------------------

class HandIndex(IntEnum)    : 
    WRIST                   = 0
    THUMB_CMC               = 1
    THUMB_MCP               = 2
    THUMB_IP                = 3
    THUMP_TIP               = 4
    INDEX_FINGER_MCP        = 5
    INDEX_FINGER_PIP        = 6
    INDEX_FINGER_DIP        = 7
    INDEX_FINGER_TIP        = 8
    MIDDLE_FINGER_MCP       = 9
    MIDDLE_FINGER_PIP       = 10
    MIDDLE_FINGER_DIP       = 11
    MIDDLE_FINGER_TIP       = 12
    RING_FINGER_MCP         = 13
    RING_FINGER_PIP         = 14
    RING_FINGER_DIP         = 15
    RING_FINGER_TIP         = 16
    PINKY_MCP               = 17
    PINKY_PIP               = 18
    PINKY_DIP               = 19
    PINKY_TIP               = 20


class PoseIndex(IntEnum):
    NOSE              = 0
    LEFT_EYE_INNER    = 1
    LEFT_EYE          = 2
    LEFT_EYE_OUTER    = 3
    RIGHT_EYE_INNER   = 4
    RIGHT_EYE         = 5
    RIGHT_EYE_OUTER   = 6
    LEFT_EAR          = 7
    RIGHT_EAR         = 8
    MOUTH_LEFT        = 9
    MOUTH_RIGHT       = 10
    LEFT_SHOULDER     = 11
    RIGHT_SHOULDER    = 12
    LEFT_ELBOW        = 13
    RIGHT_ELBOW       = 14
    LEFT_WRIST        = 15
    RIGHT_WRIST       = 16
    LEFT_PINKY        = 17
    RIGHT_PINKY       = 18
    LEFT_INDEX        = 19
    RIGHT_INDEX       = 20
    LEFT_THUMB        = 21
    RIGHT_THUMB       = 22
    LEFT_HIP          = 23
    RIGHT_HIP         = 24
    LEFT_KNEE         = 25
    RIGHT_KNEE        = 26
    LEFT_ANKLE        = 27
    RIGHT_ANKLE       = 28
    LEFT_HEEL         = 29
    RIGHT_HEEL        = 30
    LEFT_FOOT_INDEX   = 31
    RIGHT_FOOT_INDEX  = 32


## -----------------------------------------------------------------------------
# LandMark Result 相关数据类
@dataclass
class LandMarkData:
    """
        单个特征点的数据结构
    """
    x:float
    y:float
    z:float
    visibility:float
    presence:float


    def getPose(self):
        return [self.x, self.y ,self.z]

@dataclass
class Category:
    index:int
    score:float
    display_name:str
    category_name:str

@dataclass
class PoseLandmarksData:
    """
        人体33个特征点的归一化坐标
    """
    nose              : LandMarkData
    left_eye_inner    : LandMarkData
    left_eye          : LandMarkData
    left_eye_outer    : LandMarkData
    right_eye_inner   : LandMarkData
    right_eye         : LandMarkData
    right_eye_outer   : LandMarkData
    left_ear          : LandMarkData
    right_ear         : LandMarkData
    mouth_left        : LandMarkData
    mouth_right       : LandMarkData
    left_shoulder     : LandMarkData
    right_shoulder    : LandMarkData
    left_elbow        : LandMarkData
    right_elbow       : LandMarkData
    left_wrist        : LandMarkData
    right_wrist       : LandMarkData
    left_pinky        : LandMarkData
    right_pinky       : LandMarkData
    left_index        : LandMarkData
    right_index       : LandMarkData
    left_thumb        : LandMarkData
    right_thumb       : LandMarkData
    left_hip          : LandMarkData
    right_hip         : LandMarkData
    left_knee         : LandMarkData
    right_knee        : LandMarkData
    left_ankle        : LandMarkData
    right_ankle       : LandMarkData
    left_heel         : LandMarkData
    right_heel        : LandMarkData
    left_foot_index   : LandMarkData
    right_foot_index  : LandMarkData

    def toList(self):
        return list(asdict(self).values())

@dataclass
class PoseWorldLandmarksData(PoseLandmarksData):
    """
        人体33个特征点的世界坐标
    """
    pass

@dataclass
class HandLandmarksData   : 
    """
        21个手部特征点的归一化坐标
    """
    wrist                 : LandMarkData                        # index = 0
    thumb_cmc             : LandMarkData                        # index = 1
    thumb_mcp             : LandMarkData                        # index = 2
    thumb_ip              : LandMarkData                        # index = 3
    thump_tip             : LandMarkData                        # index = 4
    index_finger_mcp      : LandMarkData                        # index = 5
    index_finger_pip      : LandMarkData                        # index = 6
    index_finger_dip      : LandMarkData                        # index = 7
    index_finger_tip      : LandMarkData                        # index = 8
    middle_finger_mcp     : LandMarkData                        # index = 9
    middle_finger_pip     : LandMarkData                        # index = 10
    middle_finger_dip     : LandMarkData                        # index = 11
    middle_finger_tip     : LandMarkData                        # index = 12
    ring_finger_mcp       : LandMarkData                        # index = 13
    ring_finger_pip       : LandMarkData                        # index = 14
    ring_finger_dip       : LandMarkData                        # index = 15
    ring_finger_tip       : LandMarkData                        # index = 16
    pinky_mcp             : LandMarkData                        # index = 17
    pinky_pip             : LandMarkData                        # index = 18
    pinky_dip             : LandMarkData                        # index = 19
    pinky_tip             : LandMarkData                        # index = 20


    def toList(self):
        return list(asdict(self).values())
    

    def getAllJointPoint(self):
        return [
            [self.thumb_cmc.getPose(), self.thumb_mcp.getPose(), self.thumb_ip.getPose()],
            [self.index_finger_mcp.getPose(), self.index_finger_pip.getPose(), self.index_finger_dip.getPose()],
            [self.middle_finger_mcp.getPose(), self.middle_finger_pip.getPose(), self.middle_finger_dip.getPose()],
            [self.ring_finger_tip.getPose(), self.ring_finger_mcp.getPose(), self.ring_finger_pip.getPose()],
            [self.pinky_mcp.getPose(), self.pinky_pip.getPose(), self.pinky_dip.getPose()]
        ]

@dataclass
class HandWorldLandmarksData(HandLandmarksData):
    """
        21个手部特征点的世界坐标
    """
    pass


@dataclass
class HandLandmarkerResultData:
    """
        手部特征点的检测结果
    """
    handedness:Category
    hand_landmarks:HandLandmarksData
    hand_world_landmarks:HandWorldLandmarksData


    def getKeyPointData(self, index:int) -> list:
        """
        获取关键点位数据

        Args:
            index (int): 索引顺序参考doc

        Returns:
            list: [hand_landmarks_data_for_index, hand_world_landmarks_data_for_index]
        """
        return [self.hand_landmarks.toList()[index], self.hand_world_landmarks.toList()[index]]



@dataclass
class GestureRecognizerResultData(HandLandmarkerResultData):
    """
        手势特征点检测结果
    """
    gestures:Category


@dataclass
class PoseLandmarkerResultData:
    """人体33个特征点跟踪结果
    """
    pose_landmarks          : PoseLandmarksData
    pose_world_landmarks    : PoseLandmarksData


    def getKeyPointData(self, index:int) -> list:
        """
        获取关键点位数据

        Args:
            index (int): 索引顺序参考doc

        Returns:
            list: [hand_landmarks_data_for_index, hand_world_landmarks_data_for_index]
        """
        return [self.pose_landmarks.toList()[index], self.pose_world_landmarks.toList()[index]]
## -----------------------------------------------------------------------------


class LandMarkObservable:
    """
        landark观察者基类
    """
    def __init__(self) -> None:
        self.__observers = []


    def add_observer(self, observer):
        self.__observers.append(observer)

    
    def remove_observer(self, observer):
        self.__observers.remove(observer)

    
    def notify_observers(self):
        for o in self.__observers:
            o.updata(self)





class PoseLandMarkDetector(LandMarkObservable):
    """
        姿势特征检测
    """
    def __init__(self, model_path, running_mode=_VisionRunningMode.LIVE_STREAM, num_poses=1, 
                 min_pose_detection_confidence=0.5,
                 min_pose_presence_confidence=0.5,
                 min_tracking_confidence=0.5,
                 output_segmentation_masks=True,
                 result_callback=None
                 ) -> None:
        self.options = _PoseLandmarkerOptions(base_options=_BaseOptions(model_asset_path=model_path),
                                             running_mode=running_mode,
                                             num_poses=num_poses,
                                             min_pose_detection_confidence=min_pose_detection_confidence,
                                             min_pose_presence_confidence=min_pose_presence_confidence,
                                             min_tracking_confidence=min_tracking_confidence,
                                             output_segmentation_masks=output_segmentation_masks,
                                             result_callback=self.callback)
        self.landmarker = _PoseLandmarker.create_from_options(self.options)
        self.input_image = None
        self.output_image = None
        self.input_timestamp = None
        self.input_flag = True
        self.result = None
        self._fps_handler = FPS()
        super().__init__()


    
    def detect_async(self, color_frame, depth_frame, frame_timestamp_ms=int(time.time() * 1000)):
        """
            异步检测,当上一帧图片未检测完成时,直接返回
        """
        # if self.input_flag:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_frame)
        self.landmarker.detect_async(mp_image, int(frame_timestamp_ms))    
        self.input_color_image = color_frame
        self.input_depth_image = depth_frame
        self.input_timestamp   = frame_timestamp_ms
        self.input_flag = False


    def get_input_image(self):
        return self.input_image
    

    def get_output_image(self):
        return self.output_image


    def callback(self, result: _PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        """
            异步检测回调,当检测完成后会调用
        """
        # self.result = result
        self._fps_handler.refresh()
        # print(f"Pose Detect:{self._fps_handler.fps} | {timestamp_ms}, {self.input_timestamp}")
        # self.input_flag = True
        if len(result.pose_landmarks) > 0:
            self.input_timestamp = timestamp_ms
            # print(result.pose_landmarks)
            

            # 创建数据
            pl = []
            for i in result.pose_landmarks[0]:
                # print(i)
                pl.append(LandMarkData(**i.__dict__))
            pls = PoseLandmarksData(*pl)
            pwl = []
            for i in result.pose_world_landmarks[0]:
                # print(i)
                pwl.append(LandMarkData(**i.__dict__))
            pwls = PoseWorldLandmarksData(*pwl)
            self.result = PoseLandmarkerResultData(pose_landmarks=pls, pose_world_landmarks=pwls)
            x , y, z = self.result.pose_world_landmarks.right_wrist.x, self.result.pose_world_landmarks.right_wrist.y, self.result.pose_world_landmarks.right_wrist.z
            # print(self.result.pose_world_landmarks.right_wrist)
            # print(round(x,4), round(y,4), round(z,4))
            self.output_image = draw_pose_landmarks_on_image(output_image.numpy_view(), result)
            FPS.putFPSToImage(self.output_image, self._fps_handler.fps)
            self.notify_observers()
        self.input_flag = True
        



class HandLandMarkDetector(LandMarkObservable):
    """
        手部特征点检测
    """
    def __init__(self, model_path, running_mode=_VisionRunningMode.LIVE_STREAM, num_hands=1,
                 min_hand_detection_confidence=0.5,
                 min_hand_presence_confidence=0.5,
                 min_tracking_confidence=0.5,
                 result_callback=None) -> None:
        self.options = _HandLandmarkerOptions(base_options=_BaseOptions(model_asset_path=model_path),
                                             running_mode=running_mode,
                                             num_hands=num_hands,
                                             min_hand_detection_confidence=min_hand_detection_confidence,
                                             min_hand_presence_confidence=min_hand_presence_confidence,
                                             min_tracking_confidence=min_tracking_confidence,
                                             result_callback=self.callback
                                             )
        self.landmarker = _HandLandmarker.create_from_options(self.options)
        self.input_image = None
        self.output_image = None
        self.input_timestamp = None
        self.result:HandLandmarkerResultData = None
        self._fps_handler = FPS()
        super().__init__()


    def detect_async(self, frame, frame_timestamp_ms=int(time.time() * 1000)):
        """
            异步检测,当上一帧图片未检测完成时,直接返回
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.landmarker.detect_async(mp_image, int(frame_timestamp_ms))  


    def get_thumb_indexfinger_tip_dis(self):
        """
            获取大拇指和食指之间的间距， unit:m
        """
        dis = None
        if self.result is not None:
            tx, ty, tz = self.result.hand_world_landmarks.thump_tip.x, self.result.hand_world_landmarks.thump_tip.y, self.result.hand_world_landmarks.thump_tip.z
            mx, my, mz = self.result.hand_world_landmarks.index_finger_tip.x, self.result.hand_world_landmarks.index_finger_tip.y, self.result.hand_world_landmarks.index_finger_tip.z
            dis = math.sqrt((tx - mx)**2+(ty-my)**2+(tz-mz)**2)
            return dis
        return None


    def callback(self, result: _HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        """
            异步检测回调,当检测完成后会调用
        """
        self.result = result
        self._fps_handler.refresh()
        # print(f"Hand Detect:{self._fps_handler.fps}")
        if len(result.hand_world_landmarks) > 0:
            self.input_timestamp = timestamp_ms
            thumb_tip = result.hand_world_landmarks[0][4]
            middle_finger_tip = result.hand_world_landmarks[0][8]
            
            

            # 创建数据
            hl = []
            for i in result.hand_landmarks[0]:
                hl.append(LandMarkData(**i.__dict__))
            hls = HandLandmarksData(*hl)
            hwl = []
            for i in result.hand_world_landmarks[0]:
                hwl.append(LandMarkData(**i.__dict__))
            hwls = HandWorldLandmarksData(*hwl)
            hddness = Category(**result.handedness[0][0].__dict__)
            self.result = HandLandmarkerResultData(handedness=hddness, hand_landmarks=hls, hand_world_landmarks=hwls)
            self.output_image = draw_hand_landmarks_on_image(output_image.numpy_view(), result)
            self.notify_observers()


class GestureLandMarkDetector(LandMarkObservable):
    """
        手势特征点检测
    """
    def __init__(self, model_path, running_mode=_VisionRunningMode.LIVE_STREAM, num_hands=1,
                 min_hand_detection_confidence=0.5, 
                 min_hand_presence_confidence=0.5, 
                 min_tracking_confidence=0.5, 
                 canned_gestures_classifier_options=None, 
                 custom_gestures_classifier_options=None,
                 result_callback=None) -> None:
        super().__init__()
        self.options = _GestureRecognizerOptions(base_options=_BaseOptions(model_asset_path=model_path),
                                                running_mode=running_mode,
                                                num_hands=num_hands,
                                                min_hand_detection_confidence=min_hand_detection_confidence,
                                                min_hand_presence_confidence=min_hand_presence_confidence,
                                                min_tracking_confidence=min_tracking_confidence,
                                                # canned_gestures_classifier_options=canned_gestures_classifier_options,
                                                # custom_gestures_classifier_options=custom_gestures_classifier_options,
                                                result_callback=self.callback)    
        self.landmarker = _GestureRecognizer.create_from_options(self.options)

        self.input_depth_image, self.input_color_image, self.input_timestamp, self.input_flag = None, None, None, True
        self.output_image = None
        self.result = None
        self._fps_handler = FPS()


    def get_thumb_indexfinger_tip_dis(self):
        dis = None
        if self.result is not None:
            tx, ty, tz = self.result.hand_world_landmarks.thump_tip.x, self.result.hand_world_landmarks.thump_tip.y, self.result.hand_world_landmarks.thump_tip.z
            mx, my, mz = self.result.hand_world_landmarks.index_finger_tip.x, self.result.hand_world_landmarks.index_finger_tip.y, self.result.hand_world_landmarks.index_finger_tip.z
            dis = math.sqrt((tx - mx)**2+(ty-my)**2+(tz-mz)**2)
            return dis
        return None


    def detect_async(self, color_frame, depth_frame,frame_timestamp_ms=int(time.time() * 1000)):
        """
            异步检测,当上一帧图片未检测完成时,直接返回
        """
        if self.input_flag:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_frame)
            self.landmarker.recognize_async(mp_image, int(frame_timestamp_ms))    
            self.input_color_image = color_frame
            self.input_depth_image = depth_frame
            self.input_timestamp   = frame_timestamp_ms
            self.input_flag = False


    def callback(self, result: _GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        """
            异步检测回调,当检测完成后会调用
        """
        # self.result = result
        self._fps_handler.refresh()
        # print(f"Gesture Detect:{self._fps_handler.fps} | {timestamp_ms}, {self.input_timestamp}")
        self.input_flag = True
        if len(result.hand_world_landmarks) > 0:
            self.input_timestamp = timestamp_ms
            thumb_tip = result.hand_world_landmarks[0][4]
            middle_finger_tip = result.hand_world_landmarks[0][8]
            

            
            # 创建数据
            hl = []
            for i in result.hand_landmarks[0]:
                hl.append(LandMarkData(**i.__dict__))
            hls = HandLandmarksData(*hl)
            hwl = []
            for i in result.hand_world_landmarks[0]:
                hwl.append(LandMarkData(**i.__dict__))
            hwls = HandWorldLandmarksData(*hwl)
            hddness = Category(**result.handedness[0][0].__dict__)
            gst= Category(**result.gestures[0][0].__dict__)
            self.result = GestureRecognizerResultData(handedness=hddness, gestures=gst, hand_landmarks=hls, hand_world_landmarks=hwls)
            # print(self.result.gestures.category_name)
            self.output_image = draw_hand_landmarks_on_image(output_image.numpy_view(), result)
            FPS.putFPSToImage(self.output_image, self._fps_handler.fps)
            self.notify_observers()
            # print(self.get_thumb_indexfinger_tip_dis())


