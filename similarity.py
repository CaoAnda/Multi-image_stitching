import cv2
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

def getSimilarity(imgA, imgB, method):
    if method == 'hist':
        return calc_similar(imgA, imgB)
    elif method == 'ssim':
        return compare_ssim(imgA, imgB, multichannel=True)
    return None