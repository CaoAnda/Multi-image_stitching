import cv2
from tqdm import tqdm


from picture import classPic, getDistance
from similarity import getSimilarity


def getNearestImg(point, pics):
    nearestImg = None
    distance = 1e9
    for pic in pics:
        _distance = getDistance(point, pic.mainColor)
        if _distance < distance:
            distance = _distance
            nearestImg = pic.image
    return nearestImg

def stitching_by_rgb_distance(oralImg, smallOralImg, pics, resultImg, targetNewSize):
    for x in tqdm(range(targetNewSize[0])):
        for y in range(targetNewSize[1]):
            h, w = classPic.picSize
            startX = x * h
            startY = y * w
            resultImg[startX: startX + h, startY: startY + w] = cv2.addWeighted(
                oralImg[startX: startX + h, startY: startY + w], 0.5, getNearestImg(smallOralImg[x, y], pics), 0.5, 0)

def getImgDistance(imgA, pics):
    nearestImg = None
    similarity = 0
    for pic in pics:
        _similarity = getSimilarity(imgA, pic.image, method='hist')
        if _similarity > similarity:
            similarity = _similarity
            nearestImg = pic.image
    return nearestImg

def stitching_by_SSIM(oralImg, smallOralImg, pics, resultImg, targetNewSize, fusion = True):
    for x in tqdm(range(targetNewSize[0])):
        for y in range(targetNewSize[1]):
            h, w = classPic.picSize
            startX = x * h
            startY = y * w
            if fusion:
                resultImg[startX: startX + h, startY: startY + w] = cv2.addWeighted(
                    oralImg[startX: startX + h, startY: startY + w], 0.5, getImgDistance(oralImg[startX: startX + h, startY: startY + w], pics), 0.5, 0)
            else:
                resultImg[startX: startX + h, startY: startY + w] = getImgDistance(oralImg[startX: startX + h, startY: startY + w], pics)