import random

import cv2
import numpy
def showImg(img):
    cv2.imshow(str(random.randint(1, 100)), img)
    cv2.imwrite('result.jpg', img)
    cv2.waitKey()


class classPic:
    picSize = (20, 20)
    def __init__(self, filename):
        image = cv2.imread(filename)
        border = 1
        image = cv2.resize(image, (self.picSize[0] - border * 2, self.picSize[1] - border * 2))

        self.mainColor = get_main_color(image)
        self.image = cv2.copyMakeBorder(image, border, border, border, border, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        # print(self.mainColor)
        # showImg(self.image)

def int2RGB(color):
    # int(BGR)
    RGB = []
    for i in range(3):
        RGB.append(color % 256)
        color //= 256
    return RGB

def RGB2int(RGB):
    color = 0
    for i in RGB[::-1]:
        color *= 256
        color += i
    # int(BGR)
    return color

def get_main_color(img : numpy.ndarray, blockSize = 2):
    RGBcount = dict()
    h, w, _ = img.shape
    for x in range(h):
        for y in range(w):
            point = RGB2int(img[x, y] // blockSize)
            try:
                RGBcount[point] += 1
            except:
                RGBcount[point] = 1
    maxRGB = max(RGBcount, key=RGBcount.get)
    # print(RGBcount[maxRGB])
    return int2RGB(maxRGB)

def getDistance(pointA, pointB):
    distance = 0
    for i in range(3):
        distance += (pointA[i] - pointB[i]) * (pointA[i] - pointB[i])
    return distance