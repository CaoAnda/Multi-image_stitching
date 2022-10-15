import os

import cv2
import numpy as np
from tqdm import tqdm

from picture import classPic, get_main_color, showImg, getDistance
from stitchingMethods import stitching_by_rgb_distance, stitching

targetImg_filename = './targetImg.jpg'
pics = []
sourceImagesPath = './sourceImages'
for filename in tqdm(os.listdir(sourceImagesPath), desc='读取素材图片'):
    filename = os.path.join(sourceImagesPath, filename)
    pics.append(classPic(filename))

oralImg = cv2.imread(targetImg_filename)
targetNewSize = (50, 50)
smallOralImg = cv2.resize(oralImg, targetNewSize)

oralImg = cv2.resize(oralImg, (targetNewSize[0] * classPic.picSize[0], targetNewSize[1] * classPic.picSize[1]))
resultImg = np.zeros((targetNewSize[0] * classPic.picSize[0], targetNewSize[1] * classPic.picSize[1], 3), np.uint8)

stitching_by_rgb_distance(oralImg, smallOralImg, pics, resultImg, targetNewSize, fusion=True)
# stitching(oralImg, smallOralImg, pics, resultImg, targetNewSize, fusion=True, method='ssim')
showImg(resultImg)

# for method in ['hist', 'ssim', 'phash', 'cos']:
#     for fusion in [True, False]:
#         stitching(
#             oralImg, smallOralImg, pics, resultImg, targetNewSize, fusion=fusion, method=method
#         )
#         showImg(resultImg, '{}_{}'.format(method, fusion))




