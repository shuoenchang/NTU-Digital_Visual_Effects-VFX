import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt


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


def show_match(img1, img2, matches):
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    combine = np.zeros([max(h1, h2), w1 + w2, 3], dtype=np.uint8)
    combine[:h1, :w1] = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    combine[:h2, w1:] = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.imshow(combine)

    for (point1, point2) in matches:
        # print(point1, point2)
        y1, x1 = point1
        y2, x2 = point2
        ax.plot([x1, w1 + x2], [y1, y2], marker='o', linewidth=0.5, markersize=1)
    plt.show()


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm