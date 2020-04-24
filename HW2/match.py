import numpy as np
from scipy.spatial.distance import cdist

def find_matches(desc1, desc2, thres=0.8):
    
    d1 = [desc1[i]['desc'] for i in range(len(desc1))]
    d2 = [desc2[i]['desc'] for i in range(len(desc2))]
    distances = cdist(d1, d2)
    sort = np.argsort(distances, axis=1)
    
    matches = []
    for i, sortIndex in enumerate(sort): # i means point 1
        first = distances[i, sortIndex[0]]
        second = distances[i, sortIndex[1]]
        if first / second < thres:
            matches.append([desc1[i]['point'], desc2[sortIndex[0]]['point']])
    return matches


def ransac(matches, iterCount=1000, threshold=3):
    maxInliner = -1
    for i in range(iterCount):
        randint = np.random.randint(0, len(matches))
        dyx = np.subtract(matches[randint][0], matches[randint][1])
        matches = np.array(matches)
        afterShift = matches[:, 0] - dyx
        diff = matches[:, 1] - afterShift
        inliner = 0
        for d in diff:
            y, x = d
            if np.sqrt(x**2+y**2)<threshold:
                inliner += 1
        if maxInliner < inliner:
            maxInliner = inliner
            bestdyx = tuple(dyx)
    return bestdyx


def combine_matches(img1, img2, dyx):
    dy, dx = dyx
    # if dx<0:
    #     img1, img2 = img2, img1
    #     dy, dx = -dy, -dx
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    print(h1,w1,h2,w2)
    print(dy, dx)
    combine = np.zeros([max(h1, h2)+abs(dy), w1+dx, 3], dtype=np.uint8) + 0
    occlusion = w2-dx
    if dy>0:
        combine[:h1, :w1-occlusion] = img1[:h1, :w1-occlusion]
        combine[dy:h2+dy, w1:] = img2[:h2, w2-dx:]
        
        for i, x in enumerate(range(w1-occlusion, w1)):  # Blending
            combine[:h1, x] += (img1[:h1, x]*((occlusion-i)/occlusion)).astype(np.uint8)
            combine[dy:h2+dy, x] += (img2[:h2, i]*(i/occlusion)).astype(np.uint8)

    else:
        combine[dy:h1+dy, :w1-occlusion] = img1[:h1, :w1-occlusion]
        combine[:h2, w1:] = img2[:h2, w2-dx:]

        for i, x in enumerate(range(w1-occlusion, w1)):  # Blending
            combine[dy:h1+dy, x] += (img1[:h1, x]*((occlusion-i)/occlusion)).astype(np.uint8)
            combine[:h2, x] += (img2[:h2, i]*(i/occlusion)).astype(np.uint8)
    
    return combine