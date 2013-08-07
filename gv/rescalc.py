from __future__ import division
import numpy as np

def calc_precision_recall(detections, tp_fn):
    indices = np.where(detections['correct'] == True)[0]
    N = indices.size
    #precisions = np.zeros(N)
    #recalls = np.zeros(N)
    precisions = []
    recalls = []
    last_i = None
    for i in xrange(N):
        indx = indices[i]
        if i==0 or indices[i-1] != indx-1:
            arr = detections['correct'][indx:]
            tp = arr.sum()
            tp_fp = arr.size
        
            recalls.append(tp / tp_fn)
            precisions.append(tp / tp_fp)
    
    if len(precisions) >= 1:
        precisions.append(precisions[-1])
    else:
        precisions.append(0)
    recalls.append(0)
    return np.asarray(precisions[::-1]), np.asarray(recalls[::-1])

def calc_ap(p, r):
    return np.trapz(p, r)
