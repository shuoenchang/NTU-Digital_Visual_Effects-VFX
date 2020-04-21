import cv2
import copy
import numpy as np


def show_image(img):
    cv2.namedWindow('My Image', cv2.WINDOW_NORMAL)
    cv2.imshow('My Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_heatimage(img):
    heatmap = None
    heatmap = cv2.normalize(img, heatmap, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cv2.namedWindow('My Image', cv2.WINDOW_NORMAL)
    cv2.imshow('My Image', heatmap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_feature(img, point):
    img_feature = copy.deepcopy(img)
    for y, x in point:
        cv2.circle(img_feature, (x, y), radius=1, color=[0, 0, 255], lineType=1)
    cv2.namedWindow('My Image', cv2.WINDOW_NORMAL)
    cv2.imshow('My Image', img_feature)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def rotate_image(image, theta, center=None):
    h, w, _ = image.shape
    if center==None:
        center = (w/2, h/2)
    M = cv2.getRotationMatrix2D(center, theta, 1)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm