
















from threading import Thread
import time
import cv2
import numpy as np

from utils.fps import FPS
from utils.realsense import RealSense
from utils.landmark import GestureLandMarkDetector, HandLandMarkDetector ,PoseLandMarkDetector
from utils.observer import GestureObserver


class BodyObserver:


    def __init__(self, ) -> None:
        pass

    

    def updata(self, obj:PoseLandMarkDetector):
        if obj.result is not None:
            print(obj.result.pose_world_landmarks.right_wrist)





# class Trigger 








if __name__ == "__main__":


    realSense = RealSense(framerate=60)
    # handMarker = HandLandMark(model_path="model/hand_landmarker.task")
    # poseMarker = PoseLandMarkDetector(model_path="model/pose_landmarker_lite.task")
    poseMarker = PoseLandMarkDetector(model_path="model/pose_landmarker_heavy.task")
    gestureMarker = GestureLandMarkDetector(model_path="model/gesture_recognizer.task")

    gestureObs = GestureObserver()
    gestureObs.register_callback(GestureObserver.FuncNameLists.INCREASE, lambda:print("inc"), duration=2, volatuationData=0.04)
    gestureObs.register_callback(GestureObserver.FuncNameLists.REDUCE, lambda:print("dec"), duration=2, volatuationData=0.04)


    bodyObs = BodyObserver()


    # gestureMarker.add_observer(gestureOsb)
    gestureObsThread = Thread(target=gestureObs.updata, args=(gestureMarker,), daemon=True, name="")
    gestureObsThread.start()

    bodyObsThread = Thread(target=bodyObs.updata, args=(gestureMarker,), daemon=True, name="")
    bodyObsThread.start()

    fps = FPS()
    timestamp = int(time.time() * 1000)
    try:
        while True:
            fps.refresh()
            depth_image, color_image = realSense.get_frame()

            # print(time.time()* 1000)
            # handMarker.detect_async(color_image, timestamp)
            poseMarker.detect_async(color_image, timestamp)
            gestureMarker.detect_async(color_image, timestamp)

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
            FPS.putFPSToImage(images, fps.fps)

            # if handMarker.output_image is not None:
            #     cv2.imshow("hand", handMarker.output_image)
            if poseMarker.output_image is not None:
                cv2.imshow("pose", poseMarker.output_image)
                pass
            if gestureMarker.output_image is not None:
                cv2.imshow("gesture", gestureMarker.output_image)
                pass

                # images = np.vstack((images,np.hstack((poseMarker.output_image, gestureMarker.output_image))))
                # images = cv2.resize(images, None, fx=0.5, fy=0.5)
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
            timestamp += 1

    finally:

        # Stop streaming
        realSense.pipeline.stop()