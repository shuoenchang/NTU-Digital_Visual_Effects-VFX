import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def read_image(dirname):
    images = []
    focals = []

    for filename in np.sort(os.listdir(dirname)):
        if os.path.splitext(filename)[1].lower() in ['.jpg', '.png']:
            imgPath = os.path.join(dirname, filename)
            im = cv2.imread(imgPath)
            images += [im]

    with open(os.path.join(dirname, 'pano.txt'), 'r') as f:
        lines = f.readlines()
    for i in range(1, len(lines)-1):
        if lines[i-1]=='\n' and lines[i+1]=='\n':
            focals += [np.float(lines[i])]
    return images, focals


def reshape_images(images, reshapeRatio=1):
    for i in range(0, len(images)):
        h, w, _ = images[i].shape
        images[i] = cv2.resize(images[i], (w//reshapeRatio, h//reshapeRatio))
    return images


def rotate_image(image, theta, center=None):
    h, w, _ = image.shape
    if center==None:
        center = (w/2, h/2)
    M = cv2.getRotationMatrix2D(center, theta, 1)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated