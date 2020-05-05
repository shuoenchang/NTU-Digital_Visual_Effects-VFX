import cv2
import numpy as np
from utils import *
def End2endAlignment(image, shift):

    # print(shift)
    total_dy = shift[0]
    # print(total_dy)
    width = image.shape[1]
    dy = np.linspace(0, total_dy, width, dtype=int)

    align = image.copy()
    for w in range(width):
        align[:,w] = np.roll(image[:,w], -dy[w], axis=0)
    
    return align

def crop(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)

    # print(np.size(img_thresh[0]))
    # print(np.size(np.where(img_thresh[0] != 0)))
    h, w, _ = image.shape
    low = 0
    upper = h-1
    for i in range(h):
        print(np.size(np.where(img_thresh[i] != 0)))
        if np.size(np.where(img_thresh[i] != 0)) > 0.9*w:
            low = i
            break
    for i in range(h-1,-1,-1):
        if np.size(np.where(img_thresh[i] != 0)) > 0.9*w:
            upper = i+1
            break
    
    print(low,upper)
    return image[low:upper]

        

if __name__ == '__main__':
     image = cv2.imread('result/parrington_align.png')
     result = crop(image)
     show_image(result)
     cv2.imwrite('result/parrington_crop.png',result)