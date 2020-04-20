import numpy as np
import cv2
from scipy import ndimage as ndi
from utils import *


def harris_detector(img, k=0.04, thresRatio=0.01):
    # Compute x and y derivatives of image
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # gray image
    I = cv2.GaussianBlur(grayImg, (5,5), 0)
    Iy, Ix = np.gradient(I)

    # Compute products of derivatives at every pixel
    Ix2 = Ix*Ix
    Iy2 = Iy*Iy
    Ixy = Ix*Iy

    # Compute the sums of the products of derivatives at each pixel
    Sx2 = cv2.GaussianBlur(Ix2, (5,5), 0)
    Sy2 = cv2.GaussianBlur(Iy2, (5,5), 0)
    Sxy = cv2.GaussianBlur(Ixy, (5,5), 0)

    # Compute the response of the detector at each pixel
    """  M = [Sx2 Sxy]
             [Sxy Sy2]  """
    detM = Sx2*Sy2 - Sxy*Sxy
    traceM = Sx2 + Sy2
    R = detM - k*(traceM**2)

    # Threshold on value of R and local maximum
    threshold = thresRatio*np.max(R)
    R[R<threshold] = 0
    localMaxR = ndi.maximum_filter(R, size=5, mode='constant')
    R[R<localMaxR] = 0
    # show_heatimage(R)
    point = np.where(R>0)
    point = np.array(point).T  # (2, n) => (n, 2)
    point[:,[0, 1]] = point[:,[1, 0]]  # (y, x) => (x, y)
    return point