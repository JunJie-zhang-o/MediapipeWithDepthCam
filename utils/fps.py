import time

import cv2


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
