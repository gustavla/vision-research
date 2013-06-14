from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='Test response of model')
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')
parser.add_argument('results', metavar='<file>', nargs=1, type=argparse.FileType('rb'), help='Filename of results file')
parser.add_argument('--limit', nargs=1, type=int, default=[None])
#parser.add_argument('output_dir', metavar='<output folder>', nargs=1, type=str, help="Output path")
#parser.add_argument('img_id', metavar='<image id>', type=int, help='ID of image in VOC repository')

args = parser.parse_args()
model_file = args.model
results_file = args.results[0]
limit = args.limit[0]
#output_dir = args.output_dir[0]
#img_id = args.img_id

import matplotlib.pylab as plt
import gv
import numpy as np
import amitgroup as ag
from plotting import plot_image
import os

detector = gv.Detector.load(model_file)

data = np.load(results_file)

#p, r = data['precisions'], data['recalls']
detections = data['detections']
detections.sort(order='confidence')
detections = detections[::-1]

# TODO:
try:
    contest = str(data['contest'])
    obj_class = data['obj_class']
except KeyError:
    contest = 'voc'
    obj_class = 'car'

psize = detector.settings['subsample_size']
radii = detector.settings['spread_radii']

kernels = detector.prepare_kernels(None, settings=dict(spread_radii=radii, subsample_size=psize))
bkg = detector.bkg_model(None, spread=True)
eps = detector.settings['min_probability']
bkg = np.clip(bkg, eps, 1 - eps)

for i, det in enumerate(detections[:limit]):
    bb = (det['top'], det['left'], det['bottom'], det['right'])
    k = det['mixcomp']
    bbobj = gv.bb.DetectionBB(bb, score=det['confidence'], confidence=det['confidence'], mixcomp=k, correct=det['correct'])

    img_id = det['img_id']
    fileobj = gv.datasets.load_file(contest, img_id, obj_class=obj_class)

    im = gv.img.load_image(fileobj.path) 
    im = gv.img.asgray(im)
    im = gv.img.resize_with_factor_new(im, 1/det['scale'])

    kern = kernels[k]

    d0, d1 = kern.shape[:2] 
    
    feats = detector.descriptor.extract_features(im, dict(spread_radii=radii, preserve_size=True))
    feats = gv.sub.subsample(feats, psize) 

    i0, j0 = det['index_pos0'], det['index_pos1'] 
    pad = max(-min(0, i0), -min(0, j0), max(0, i0+d0 - feats.shape[0]), min(0, j0+d1 - feats.shape[1]))

    feats = ag.util.zeropad(feats, (pad, pad, 0))

    i0, j0 = det['index_pos0'], det['index_pos1'] 
    pad = max(-min(0, i0), -min(0, j0), max(0, i0+d0 - feats.shape[0]), min(0, j0+d1 - feats.shape[1]))

    feats = ag.util.zeropad(feats, (pad, pad, 0))

    weights = np.log(kern / (1 - kern) * ((1 - bkg) / bkg))

    #X = feats_pad[pad+i0-d0//2:pad+i0-d0//2+d0, pad+j0-d1//2:pad+j0-d1//2+d1]
    X = feats[pad+i0:pad+i0+d0, pad+j0:pad+j0+d1]

    R = (X * weights).sum()
    Rst = (R - detector.fixed_train_mean[k]) / detector.fixed_train_std[k]
    
    # Replace bounding boxes with this single one
    fileobj.boxes[:] = [bbobj]
    
    fn = 'det-{0}.png'.format(i)
    print '{0}: {1:.2f}'.format(fn, Rst)
    #print 'scale', np.log(det['scale'])/np.log(2)
    #print i0, j0
