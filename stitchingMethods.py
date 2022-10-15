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

def stitching_by_rgb_distance(oralImg, smallOralImg, pics, resultImg, targetNewSize, fusion = True):
    for x in tqdm(range(targetNewSize[0])):
        for y in range(targetNewSize[1]):
            h, w = classPic.picSize
            startX = x * h
            startY = y * w
            if fusion:
                resultImg[startX: startX + h, startY: startY + w] = cv2.addWeighted(
                    oralImg[startX: startX + h, startY: startY + w], 0.5, getNearestImg(smallOralImg[x, y], pics), 0.5, 0)
            else:
                resultImg[startX: startX + h, startY: startY + w] = getNearestImg(smallOralImg[x, y], pics)

def getImgDistance(imgA, pics, method):
    nearestImg = None
    similarity = 0
    for pic in pics:
        _similarity = getSimilarity(imgA.copy(), pic.image.copy(), method=method)
        if _similarity > similarity:
            similarity = _similarity
            nearestImg = pic.image
    return nearestImg

def stitching(oralImg, smallOralImg, pics, resultImg, targetNewSize, fusion = True, method = None):
    for x in tqdm(range(targetNewSize[0]), desc='fusion:%s, method:%s'%(fusion, method)):
        for y in range(targetNewSize[1]):
            h, w = classPic.picSize
            startX = x * h
            startY = y * w
            if fusion:
                resultImg[startX: startX + h, startY: startY + w] = cv2.addWeighted(
                    oralImg[startX: startX + h, startY: startY + w], 0.5, getImgDistance(oralImg[startX: startX + h, startY: startY + w], pics, method), 0.5, 0)
            else:
                resultImg[startX: startX + h, startY: startY + w] = getImgDistance(oralImg[startX: startX + h, startY: startY + w], pics, method)