import cv2
import numpy as np
IMG_SIZE = (64, 64)
_hog = cv2.HOGDescriptor(
    _winSize=(64, 64), _blockSize=(16, 16),
    _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9
)
def extract_features(bgr_image):
    """Turn one BGR image into a 1D feature vector (HOG + color histogram)."""
    img = cv2.resize(bgr_image, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_feat = _hog.compute(gray).flatten()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = []
    for ch, rng in zip(range(3), [(0, 180), (0, 256), (0, 256)]):
        h = cv2.calcHist([hsv], [ch], None, [16], rng).flatten()
        h = h / (h.sum() + 1e-6)          
        hist.append(h)
    hist_feat = np.concatenate(hist)
    return np.concatenate([hog_feat, hist_feat]).astype("float32")