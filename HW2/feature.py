import numpy as np
import cv2
from scipy.ndimage import maximum_filter
from utils import *
from images import *


def compute_gradient(image):
    grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # gray image
    I = cv2.GaussianBlur(grayImg, (5,5), 0)
    Iy, Ix = np.gradient(I)
    return Ix, Iy


def harris_detector(image, k=0.04, thresRatio=0.01):
    # Compute x and y derivatives of image
    Ix, Iy = compute_gradient(image)

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
    localMaxR = maximum_filter(R, size=3, mode='constant')
    R[R<localMaxR] = 0
    # show_heatimage(R)
    point = np.where(R>0)
    point = np.array(point).T  # (2, n) => (n, 2)
    # point[:,[0, 1]] = point[:,[1, 0]]  # (y, x) => (x, y)
    return point


def orientation_histogram(image, bins, sigma):
    Ix, Iy = compute_gradient(image)
    magnitude = np.sqrt(Ix**2 + Iy**2)
    theta = np.arctan2(Iy, Ix)*180/np.pi
    theta[theta<0] = theta[theta<0]+360
    
    binSize = 360/bins
    bucket = np.round(theta/binSize)
    histogram = np.zeros((bins,) + magnitude.shape)  # (bins, h, w)
    for b in range(bins):
        histogram[b][bucket==b] = 1
        histogram[b] *= magnitude
        histogram[b] = cv2.GaussianBlur(histogram[b], (5,5), sigma)
    
    return histogram


def keypoint_descriptor(image, keyPoints):
    # Orientation assignment
    histogram = orientation_histogram(image, bins=36, sigma=1.5)
    orientations = np.argmax(histogram, axis=0)*10 + 5
    
    # Local image descriptor
    descriptors = []
    h, w, _ = image.shape
    for y, x in keyPoints:
        rotated = rotate_image(image, orientations[y, x], (x,y))
        histogram = orientation_histogram(rotated, bins=8, sigma=8)
        if y-8>0 and y+8<h and x-8>0 and x+8<w:  # else discard this keypoint
            # print(x, y)
            desc = []
            for subY in range(y-8, y+8, 4):
                for subX in range(x-8, x+8, 4):
                    subHistogram = []
                    for bin in range(8):
                        subHistogram.append(np.sum(histogram[bin][subY:subY+4, subX:subX+4]))
                    desc += subHistogram
            desc = normalize(desc)
            if np.any(desc>0.2):
                desc[desc>0.2] = 0.2
                desc = normalize(desc)
            descriptors.append({'point':(y, x), 'desc':desc})
    return descriptors