from __future__ import print_function
from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='Test response of model')
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')
parser.add_argument('output', metavar='<output file>', type=argparse.FileType('wb'), help='Filename of output file')
parser.add_argument('--limit', nargs=1, type=int, default=[None])
parser.add_argument('--mini', action='store_true', default=False)

args = parser.parse_args()
model_file = args.model
output_file = args.output
limit = args.limit[0]
mini = args.mini

import gv
import amitgroup as ag
import numpy as np
import scipy.integrate

from config import VOCSETTINGS

detector = gv.Detector.load(model_file)

dataset = ['test', 'train'][mini]
files, tot = gv.voc.load_files(VOCSETTINGS, 'bicycle', dataset=dataset)

tot_tp = 0
tot_tp_fp = 0
tot_tp_fn = 0

detections = []

for fileobj in files[:limit]:
    img = gv.img.load_image(fileobj.path)
    grayscale_img = img.mean(axis=-1)
    if mini and len(fileobj.boxes) == 0:
        continue

    tp = tp_fp = tp_fn = 0

    # Count tp+fn
    for bbobj in fileobj.boxes:
        if not bbobj.difficult:
            tp_fn += 1 

    bbs = detector.detect_coarse(grayscale_img, fileobj=fileobj)

    tp_fp += len(bbs)
    
    for bbobj in bbs:
        detections.append((bbobj.confidence, bbobj.correct))
        if bbobj.correct and not bbobj.difficult:
            tp += 1

    print("Testing file {0} (tp:{1} tp+fp:{2} tp+fn:{3}".format(fileobj.img_id, tp, tp_fp, tp_fn))

    tot_tp += tp
    tot_tp_fp += tp_fp
    tot_tp_fn += tp_fn

detections = np.array(detections, dtype=[('confidence', float), ('correct', bool)])
detections.sort(order='confidence')

def calc_precision_recall(detections, tp_fn):
    indices = np.where(detections['correct'] == True)[0]
    N = indices.size
    precisions = np.zeros(N)
    recalls = np.zeros(N)
    for i in xrange(N):
        indx = indices[i]
        arr = detections['correct'][indx:]
        tp = arr.sum()
        tp_fp = arr.size
    
        recalls[-1-i] = tp / tp_fn
        precisions[-1-i] = tp / tp_fp
    
    return precisions, recalls

precisions, recalls = calc_precision_recall(detections, tot_tp_fn)
ap = scipy.integrate.trapz(precisions, recalls)
np.savez(output_file, precisions=precisions, recalls=recalls, detections=detections, tp_fn=tot_tp_fn, ap=ap)

print('tp', tot_tp)
print('tp+fp', tot_tp_fp)
print('tp+fn', tot_tp_fn)
print('----------------')
if tot_tp_fp:
    print('Precision', tot_tp / tot_tp_fp)
if tot_tp_fn:
    print('Recall', tot_tp / tot_tp_fn)
print('AP', ap)
