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