"""
    手势检测,LIVE_STREAM mode,异步接收
"""


import math
import time
import mediapipe as mp
from mediapipe.tasks import python
# from mediapipe.tasks.python import version
import cv2
import os

os.environ["MEDIAPIPE_GPU_ENABLED"] = "1"
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

class FPS:

    def __init__(self) -> None:
        self.startT = time.time()        
        self.frameCount = 0
        self._fps = 0

    
    @property
    def fps(self):
        return self._fps


    def refresh(self):
        self.frameCount += 1
        now = time.time()
        if now - self.startT > 1:
            self._fps = int(self.frameCount / (now - self.startT))
            self.frameCount = 0
            self.startT = now


    @classmethod
    def putFPSToImage(cls, img, fps):
        # 添加文字到图像左上角
        text = f"FPS:{int(fps)}"
        org = (10, 30)  # 文字起始位置
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (0, 0, 255)  
        thickness = 2

        cv2.putText(img, text, org, font, fontScale, color, thickness, cv2.LINE_AA)


cap  = cv2.VideoCapture(0)
time.sleep(1)
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult



def draw_landmarks_on_image(rgb_image, detection_result):
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



global result
gresult = None
fps2 = FPS()
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    # print('hand landmarker result: {}'.format(result))
    fps2.refresh()
    print(fps2.fps)
    global gresult
    gresult = result
    if len(result.hand_world_landmarks) > 0:
        # print(result.handedness)
        thumb_tip = result.hand_world_landmarks[0][4]
        middle_finger_tip = result.hand_world_landmarks[0][8]
        print(thumb_tip)
        print(middle_finger_tip)
        tx, ty, tz = thumb_tip.x, thumb_tip.y, thumb_tip.z
        mx, my, mz = middle_finger_tip.x, middle_finger_tip.y, middle_finger_tip.z
        dis = math.sqrt((tx - mx)**2+(ty-my)**2+(tz-mz)**2)
        print(dis)


    # annotated_image = draw_landmarks_on_image(output_image, result)
    # showImage = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    # cv2.imshow("annotated_image", showImage)


VisionRunningMode = mp.tasks.vision.RunningMode
# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='./medel/hand_landmarker.task')
options = vision.HandLandmarkerOptions(running_mode=VisionRunningMode.LIVE_STREAM,
                                       base_options=base_options,
                                       num_hands=2,
                                       result_callback=print_result)
detector = vision.HandLandmarker.create_from_options(options)

# STEP 3: Load the input image.
# image = mp.Image.create_from_file("image.jpg")

# cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
fps = FPS()
frame_timestamp_ms = int(time.time())
while 1:

    _, frame = cap.read()
    fps.refresh()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    # STEP 4: Detect hand landmarks from the input image.
    # detection_result = detector.detect(mp_image)
    detector.detect_async(mp_image, int(frame_timestamp_ms))

    
    showImage = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


    FPS.putFPSToImage(showImage, fps.fps)
    cv2.imshow("img", showImage)
    
    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC hit
        print("Escape hit, closing...")
        break
    frame_timestamp_ms +=1

cap.release()
cv2.destroyAllWindows()


print("--")


