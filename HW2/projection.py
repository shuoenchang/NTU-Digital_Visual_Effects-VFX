import numpy as np
import math
import cv2

def cylindrical_projection(images, focals):
    projection = np.zeros( (len(focals),) + images[0].shape, dtype=np.uint8)
    print( (len(focals),) + images[0].shape )
    
    h, w, _ = images[0].shape
    for i in range(h):
        for j in range(w):
            x = j - int(w/2)
            y = h-1 - i
            x1 = [int(focal*np.arctan(x/focal))+ int(w/2) for focal in focals]
            y1 = [h-1 - int(focal*y/math.sqrt(focal*focal+x*x)) for focal in focals]
            # print("(i,j):(",i,j,"), (i',j'):(",y1[0],x1[0],"), pixels: (",images[0][i,j])
            for f in range(len(focals)):
                # print("(i,j):(",i,j,"), (i',j'):(",y1[f],x1[f],"), pixels: (",images[f][i,j])
                projection[f][y1[f],x1[f]] = images[f][i,j]
    
    for image in projection:
        cv2.imshow('projection',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return projection
