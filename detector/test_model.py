from __future__ import print_function
from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='Test response of model')
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')
parser.add_argument('obj_class', metavar='<object class>', type=str, help='Object class')
parser.add_argument('output', metavar='<output file>', type=argparse.FileType('wb'), help='Filename of output file')
parser.add_argument('--limit', nargs=1, type=int, default=[None])
parser.add_argument('--mini', action='store_true', default=False)
parser.add_argument('--contest', type=str, choices=('voc', 'uiuc', 'uiuc-multiscale'), default='voc', help='Contest to try on')

args = parser.parse_args()
model_file = args.model
obj_class = args.obj_class
output_file = args.output
limit = args.limit[0]
mini = args.mini
contest = args.contest

import gv
import amitgroup as ag
import numpy as np
import scipy.integrate
import skimage.data

from config import VOCSETTINGS

detector = gv.Detector.load(model_file)

#dataset = ['val', 'train'][mini]
#dataset = ['val', 'train'][mini]
dataset = 'val'
if contest == 'voc':
    files, tot = gv.voc.load_files(VOCSETTINGS, obj_class, dataset=dataset)
elif contest == 'uiuc':
    files, tot = gv.uiuc.load_testing_files()
elif contest == 'uiuc-multiscale':
    files, tot = gv.uiuc.load_testing_files(single_scale=False)

tot_tp = 0
tot_tp_fp = 0
tot_tp_fn = 0

detections = []

if mini:
    files = filter(lambda x: len(x.boxes) > 0, files)
files = files[:limit]

#fout = open("detections.txt", "w")

def detect(fileobj):
    detections = []
    img = gv.img.load_image(fileobj.path)
    grayscale_img = gv.img.asgray(img)
    
    # TODO: Experimental
    grayscale_img = ag.util.blur_image(grayscale_img, 0.05)

    tp = tp_fp = tp_fn = 0

    # Count tp+fn
    for bbobj in fileobj.boxes:
        if not bbobj.difficult:
            tp_fn += 1 

    bbs = detector.detect_coarse(grayscale_img, fileobj=fileobj)

    tp_fp += len(bbs)
    
    for bbobj in bbs:
        #print("{0:06d} {1} {2} {3} {4} {5}".format(fileobj.img_id, bbobj.confidence, int(bbobj.box[0]), int(bbobj.box[1]), int(bbobj.box[2]), int(bbobj.box[3])), file=fout)
        detections.append((bbobj.confidence, bbobj.score0, bbobj.score1, bbobj.plusscore, bbobj.correct, bbobj.mixcomp, fileobj.img_id, int(bbobj.box[1]), int(bbobj.box[0]), int(bbobj.box[3]), int(bbobj.box[2])))
        #fout.flush()
        if bbobj.correct and not bbobj.difficult:
            tp += 1

    print("Testing file {0} (tp:{1} tp+fp:{2} tp+fn:{3})".format(fileobj.img_id, tp, tp_fp, tp_fn))

    return (tp, tp_fp, tp_fn, detections)


if 1:
    from multiprocessing import Pool
    p = Pool(7)
    mapf = p.map
else:
    mapf = map

res = mapf(detect, files)


for tp, tp_fp, tp_fn, dets in res:
    tot_tp += tp
    tot_tp_fp += tp_fp
    tot_tp_fn += tp_fn
    detections.extend(dets)

detections = np.array(detections, dtype=[('confidence', float), ('score0', float), ('score1', float), ('plusscore', float), ('correct', bool), ('mixcomp', int), ('img_id', int), ('left', int), ('top', int), ('right', int), ('bottom', int)])
detections.sort(order='confidence')

p, r = gv.rescalc.calc_precision_recall(detections, tot_tp_fn)
ap = gv.rescalc.calc_ap(p, r) 
np.savez(output_file, detections=detections, tp_fn=tot_tp_fn, ap=ap)

print('tp', tot_tp)
print('tp+fp', tot_tp_fp)
print('tp+fn', tot_tp_fn)
print('----------------')
#if tot_tp_fp:
#    print('Precision', tot_tp / tot_tp_fp)
#if tot_tp_fn:
#    print('Recall', tot_tp / tot_tp_fn)
print('AP', ap)
