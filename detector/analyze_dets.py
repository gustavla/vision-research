from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='Test response of model')
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')
parser.add_argument('results', metavar='<file>', type=argparse.FileType('rb'), help='Filename of results file')
parser.add_argument('--limit', type=int, default=None)
#parser.add_argument('output_dir', metavar='<output folder>', nargs=1, type=str, help="Output path")
#parser.add_argument('img_id', metavar='<image id>', type=int, help='ID of image in VOC repository')

args = parser.parse_args()
model_file = args.model
results_file = args.results
limit = args.limit
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
all_bkg = detector.bkg_model(None, spread=True)
eps = detector.settings['min_probability']
#bkg = np.clip(bkg, eps, 1 - eps)

analyze_mixcomp = -1 
Y = np.zeros((2,) + kernels[analyze_mixcomp].shape, dtype=np.uint32)
totals = np.zeros(2, dtype=np.uint32)

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
    bkg = all_bkg[k]
    kern = np.clip(kern, eps, 1 - eps)
    bkg = np.clip(bkg, eps, 1 - eps)

    d0, d1 = kern.shape[:2] 
    
    feats = detector.descriptor.extract_features(im, dict(spread_radii=radii, subsample_size=psize, preserve_size=False))

    i0, j0 = det['index_pos0'], det['index_pos1'] 
    pad = max(-min(0, i0), -min(0, j0), max(0, i0+d0 - feats.shape[0]), max(0, j0+d1 - feats.shape[1]))

    feats = ag.util.zeropad(feats, (pad, pad, 0))

    weights = np.log(kern / (1 - kern) * ((1 - bkg) / bkg))

    #X = feats_pad[pad+i0-d0//2:pad+i0-d0//2+d0, pad+j0-d1//2:pad+j0-d1//2+d1]
    #print pad, i0, d0, j0, d1, 'abc', j0+d1 - feats.shape[1]
    #print 'feats', feats.shape
    X = feats[pad+i0:pad+i0+d0, pad+j0:pad+j0+d1]
    #print 'X', X.shape

    if analyze_mixcomp == k or analyze_mixcomp == -1:
        Y[det['correct']] += X
        totals[det['correct']] += 1

    R = float((X * weights).sum())
    #Rst = (R - detector.fixed_train_mean[k]) / detector.fixed_train_std[k]
    
    from gv.fast import nonparametric_rescore
    Rarray = R * np.ones((1, 1))
    info = detector.standardization_info[k]
    nonparametric_rescore(Rarray, info['start'], info['step'], info['points'])
    #Rst = R
    Rst = Rarray[0,0]
    
    # Replace bounding boxes with this single one
    fileobj.boxes[:] = [bbobj]
    
    fn = 'det-{0}.png {1}'.format(i, img_id)
    print '{3} {0}: {1:.2f} {2:.2f} ({4})'.format(fn, Rst, det['confidence'], ['X', '.'][det['correct']], det['mixcomp'])
    #print 'scale', np.log(det['scale'])/np.log(2)
    #print i0, j0

Z = Y.astype(np.float64) / totals.reshape((-1,) + (1,)*(Y.ndim-1))
G = (Z * weights)

if 1: 
    Gmeans = [G[i].mean(axis=-1) for i in xrange(2)]
    m = max(np.fabs(Gmeans[0]).max(), np.fabs(Gmeans[1]).max())

    plt.subplot(211)
    plt.imshow(Gmeans[0], interpolation='nearest', vmin=-m, vmax=m, cmap=plt.cm.RdBu_r)
    plt.title('FP')
    plt.subplot(212)
    plt.imshow(Gmeans[1], interpolation='nearest', vmin=-m, vmax=m, cmap=plt.cm.RdBu_r)
    plt.title('TP')
    plt.show()

if 1:
    Zmeans = [Z[i].mean(axis=-1) for i in xrange(2)]
    m = max(np.fabs(Zmeans[0]).max(), np.fabs(Zmeans[1]).max())

    plt.subplot(211)
    plt.imshow(Zmeans[0], interpolation='nearest', vmax=m)
    plt.title('FP')
    plt.subplot(212)
    plt.imshow(Zmeans[1], interpolation='nearest', vmax=m)
    plt.title('TP')
    plt.show()

# Open a shell where we can play around with the data
import IPython
IPython.embed()
