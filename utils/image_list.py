

from collections import deque
import time


class ImageList:


    def __init__(self, length:int=60) -> None:
        
        self.color_image = deque(maxlen=length)
        self.depth_image = deque(maxlen=length)
        self.timestamp = deque(maxlen=length)



    def set_image_data(self, color_image, depth_image, timestamp):

        self.color_image.append(color_image)
        self.depth_image.append(depth_image)
        self.timestamp.append(timestamp)


    
    def get_image_data(self, timestamp):

        index = self.timestamp.index(timestamp)
        return self.color_image[index], self.depth_image[index]
    

    def get_depth_image(self, timestamp):
        index = self.timestamp.index(timestamp)
        return self.depth_image[index]
    

    def get_color_image(self, timestamp):
        index = self.timestamp.index(timestamp)
        return self.color_image[index]
    


if __name__ == "__main__":

    im = ImageList()

    i = 0
    while 1:
        time.sleep(0.001)
        im.set_image_data(i,i,i)
        print(im.timestamp[0])
        i=i+1