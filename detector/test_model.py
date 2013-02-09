from __future__ import print_function
from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='Test response of model')
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')

args = parser.parse_args()
model_file = args.model

import gv
import amitgroup as ag
import numpy as np

from config import VOCSETTINGS

detector = gv.Detector.load(model_file)

files, tot = gv.voc.load_files(VOCSETTINGS, 'bicycle', dataset='test')

tot_tp = 0
tot_tp_fp = 0
tot_tp_fn = 0

detections = []

for fileobj in files[0:50]:
    print("Testing file {0}".format(fileobj.img_id))
    img = gv.img.load_image(fileobj.path)
    grayscale_img = img.mean(axis=-1)

    tp = tp_fp = tp_fn = 0

    # Count tp+fn
    for bbobj in fileobj.boxes:
        if not bbobj.difficult:
            tp_fn += 1 

    bbs = detector.detect_coarse(grayscale_img, fileobj=fileobj)

    tp_fp += len(bbs)
    
    for bbobj in bbs:
        detections.append((bbobj.confidence, bbobj.correct))
        if bbobj.correct:
            tp += 1

    tot_tp += tp
    tot_tp_fp += tp_fp
    tot_tp_fn += tp_fn

np.save('conf.npy', detections)

def calc_precision_recall(detections, tp_fn):
    arr = np.asarray(detections)
    arr.sort(axis=0)

    N = arr.shape[0]
    precisions = np.zeros(N)
    recalls = np.zeros(N)
    
    for i in xrange(N):
        tp = arr[i:,1].sum()
        tp_fp = arr[i:,1].size
    
        recalls[i] = tp / tp_fn
        precisions[i] = tp / tp_fp
    
    return precisions, recalls

precisions, recalls = calc_precision_recall(detections, tot_tp_fn)
np.savez('conf.npz', precisions=precisions, recalls=recalls)

print('tp', tot_tp)
print('tp+fp', tot_tp_fp)
print('tp+fn', tot_tp_fn)
print('----')
if tot_tp_fp:
    print('Precision', tot_tp / tot_tp_fp)
if tot_tp_fn:
    print('Recall', tot_tp / tot_tp_fn)
