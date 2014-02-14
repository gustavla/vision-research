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

def calc_fppi_miss_rate(detections, tp_fn, num_images):
    indices = np.where(detections['correct'] == True)[0]
    N = indices.size
    #precisions = np.zeros(N)
    #recalls = np.zeros(N)
    fppi = []
    recalls = []
    last_i = None
    for i in xrange(N):
        indx = indices[i]
        if i==0 or indices[i-1] != indx-1:
            arr = detections['correct'][indx:]
            tp = arr.sum()
            tp_fp = arr.size
            fp = tp_fp - tp
        
            recalls.append(tp / tp_fn)
            fppi.append(fp / num_images)
    
    if len(fppi) >= 1:
        fppi.append(fppi[-1])
    else:
        fppi.append(0)
    recalls.append(0)
    return np.asarray(fppi[::-1]), 1-np.asarray(recalls[::-1])

def calc_fppi_summary(fppi, miss_rate):
    fppis = 10**np.linspace(-2, 0, 9)

    scores = np.zeros(len(fppis))
    for i in xrange(len(fppis)):
        ind = np.where(fppi >= fppis[i])[0][0]
        #fppis[i], miss_rate[ind]
        scores[i] = miss_rate[ind]
    return scores.mean()


# According to VOC
if 0:
    def calc_ap(p, r):
        ap = 0
        for t in np.arange(0, 1+1e-8, 11):
            ii = np.where(r >= t)[0]
            pre = 0
            if ii.size > 0:
                pre += p[ii].max()  
            
            ap += pre / 11
        return ap
