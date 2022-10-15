import cv2
import numpy as np
from numpy import average, dot
from scipy import linalg
from skimage.metrics import structural_similarity as compare_ssim

def hist_similar(lh, rh):
    assert len(lh) == len(rh)
    hist = sum(1 - (0 if l == r else float(abs(l - r)) / max(l, r)) for l, r in zip(lh, rh)) / len(lh)
    return hist

def calc_similar(imgA, imgB):
    calc_sim = hist_similar(
        cv2.calcHist(imgA, [3], None, [256], [0, 256]),
        cv2.calcHist(imgB, [3], None, [256], [0, 256])
    )
    return calc_sim

def image_similarity_vectors_via_numpy(image1, image2):
    # images = [cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY), cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)]
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image:
            vector.append(average(pixel_tuple))
        vectors.append(np.array(vector))
        # linalg=linear（线性）+algebra（代数），norm则表示范数
        # 求图片的范数？？
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    # dot返回的是点积，对二维数组（矩阵）进行计算
    res = dot(a / a_norm, b / b_norm)
    return res

# 感知哈希算法
def pHash(img):
    img = cv2.resize(img, (32, 32))
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 将灰度图转为浮点型，再进行dct变换
    dct = cv2.dct(np.float32(gray))
    # opencv实现的掩码操作
    dct_roi = dct[0:8, 0:8]

    hash = []
    avreage = np.mean(dct_roi)
    for i in range(dct_roi.shape[0]):
        for j in range(dct_roi.shape[1]):
            if dct_roi[i, j] > avreage:
                hash.append(1)
            else:
                hash.append(0)
    return hash

# 返回值为汉明距离，距离越大，相似度越低
def cmpHash(hash1, hash2):
    n = 0
    # hash长度不同则返回-1代表传参出错
    assert len(hash1) == len(hash2)
    # 遍历判断
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        if hash1[i] != hash2[i]:
            n = n + 1
    return n
def similarity_by_phash(imgA, imgB):
    return 1 / cmpHash(pHash(imgA), pHash(imgB))

'''
效果较好的：

'''
def getSimilarity(imgA, imgB, method):
    similarity = None
    if method == 'hist':
        similarity = calc_similar(imgA, imgB)
    elif method == 'ssim':
        similarity = compare_ssim(imgA, imgB, multichannel=True)
    elif method == 'phash':
        similarity = similarity_by_phash(imgA, imgB)
    elif method == 'cos':
        similarity = image_similarity_vectors_via_numpy(imgA, imgB)
    # print(similarity)
    return similarity