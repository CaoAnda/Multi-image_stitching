import os

import cv2
import numpy as np

from picture import classPic, get_main_color, showImg, getDistance
from stitchingMethods import stitching_by_rgb_distance, stitching_by_SSIM

targetImg_filename = './targetImg.jpg'
pics = []
sourceImagesPath = './sourceImages'
for filename in os.listdir(sourceImagesPath):
    filename = os.path.join(sourceImagesPath, filename)
    pics.append(classPic(filename))

oralImg = cv2.imread(targetImg_filename)
targetNewSize = (50, 50)
smallOralImg = cv2.resize(oralImg, targetNewSize)

oralImg = cv2.resize(oralImg, (targetNewSize[0] * classPic.picSize[0], targetNewSize[1] * classPic.picSize[1]))
resultImg = np.zeros((targetNewSize[0] * classPic.picSize[0], targetNewSize[1] * classPic.picSize[1], 3), np.uint8)

# stitching_by_rgb_distance(oralImg, smallOralImg, pics, resultImg, targetNewSize)
stitching_by_SSIM(oralImg, smallOralImg, pics, resultImg, targetNewSize, fusion=True)

showImg(resultImg)
