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
        if maxInliner<inliner:
            maxInliner=inliner
            bestdyx = tuple(dyx)
    return bestdyx