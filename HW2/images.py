import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable


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


def show_image(img):
    cv2.namedWindow('My Image', cv2.WINDOW_NORMAL)
    cv2.imshow('My Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_heatimage(img):
    heatmap = None
    heatmap = cv2.normalize(img, heatmap, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cv2.namedWindow('My Image', cv2.WINDOW_NORMAL)
    cv2.imshow('My Image', heatmap)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    inputPath = 'data/'+'parrington'
    images, focals = read_image(inputPath)
    images = reshape_images(images, 1)
    show_image(images[0])