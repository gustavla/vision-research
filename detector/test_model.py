from __future__ import print_function
from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='Test response of model')
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')
parser.add_argument('obj_class', metavar='<object class>', type=str, help='Object class')
parser.add_argument('output', metavar='<output file>', type=argparse.FileType('wb'), help='Filename of output file')
parser.add_argument('--limit', nargs=1, type=int, default=[None])
parser.add_argument('--mini', action='store_true', default=False)
parser.add_argument('--contest', type=str, choices=('voc-val', 'voc-profile', 'voc-easy', 'uiuc', 'uiuc-multiscale'), default='voc-val', help='Contest to try on')
parser.add_argument('--no-threading', action='store_true', default=False, help='Turn off threading')

args = parser.parse_args()
model_file = args.model
obj_class = args.obj_class
output_file = args.output
limit = args.limit[0]
mini = args.mini
threading = not args.no_threading
contest = args.contest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import gv
import amitgroup as ag
import numpy as np
import scipy.integrate
import skimage.data

detector = gv.Detector.load(model_file)

#dataset = ['val', 'train'][mini]
#dataset = ['val', 'train'][mini]
#dataset = 'val'
if contest == 'voc-val':
    files, tot = gv.voc.load_files(obj_class, dataset='val')
elif contest == 'voc-profile':
    files, tot = gv.voc.load_files(obj_class, dataset='profile')
elif contest == 'voc-easy':
    files, tot = gv.voc.load_files(obj_class, dataset='easy')
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
    
    tp = tp_fp = tp_fn = 0

    # Count tp+fn
    for bbobj in fileobj.boxes:
        if not bbobj.difficult:
            tp_fn += 1 

    bbs = detector.detect(grayscale_img, fileobj=fileobj)

    tp_fp += len(bbs)
    
    for bbobj in bbs:
        #print("{0:06d} {1} {2} {3} {4} {5}".format(fileobj.img_id, bbobj.confidence, int(bbobj.box[0]), int(bbobj.box[1]), int(bbobj.box[2]), int(bbobj.box[3])), file=fout)
        detections.append((bbobj.confidence, bbobj.scale, bbobj.score0, bbobj.score1, bbobj.plusscore, bbobj.correct, bbobj.mixcomp, fileobj.img_id, int(bbobj.box[1]), int(bbobj.box[0]), int(bbobj.box[3]), int(bbobj.box[2]), bbobj.index_pos[0], bbobj.index_pos[1]))
        #fout.flush()
        if bbobj.correct and not bbobj.difficult:
            tp += 1

    print("Testing file {0} (tp:{1} tp+fp:{2} tp+fn:{3})".format(fileobj.img_id, tp, tp_fp, tp_fn))

    return (tp, tp_fp, tp_fn, detections)


if threading:
    from multiprocessing import Pool
    p = Pool(7)
    imapf = p.imap_unordered
else:
    from itertools import imap as imapf

res = imapf(detect, files)

per_file_dets = []

for tp, tp_fp, tp_fn, dets in res:
    tot_tp += tp
    tot_tp_fp += tp_fp
    tot_tp_fn += tp_fn
    detections.extend(dets)
    per_file_dets.append(dets)


if 1:
    from operator import itemgetter
    plt.clf()
    for i, file_dets in enumerate(per_file_dets):
        scores = map(itemgetter(0), file_dets)
        corrects = map(itemgetter(5), file_dets)
        colors = map(lambda x: ['r', 'g'][x], corrects)
        plt.scatter([i+1]*len(file_dets), scores, c=colors, s=50, alpha=0.75)

    plt.savefig('detvis.png')
    
    
detections = np.array(detections, dtype=[('confidence', float), ('scale', float), ('score0', float), ('score1', float), ('plusscore', float), ('correct', bool), ('mixcomp', int), ('img_id', int), ('left', int), ('top', int), ('right', int), ('bottom', int), ('index_pos0', int), ('index_pos1', int)])
detections.sort(order='confidence')

p, r = gv.rescalc.calc_precision_recall(detections, tot_tp_fn)
ap = gv.rescalc.calc_ap(p, r) 
np.savez(output_file, detections=detections, tp_fn=tot_tp_fn, ap=ap, contest=contest, obj_class=obj_class)

print('tp', tot_tp)
print('tp+fp', tot_tp_fp)
print('tp+fn', tot_tp_fn)
print('----------------')
#if tot_tp_fp:
#    print('Precision', tot_tp / tot_tp_fp)
#if tot_tp_fn:
#    print('Recall', tot_tp / tot_tp_fn)
print('AP {0:.2f}% ({1})'.format(100*ap, ap))
