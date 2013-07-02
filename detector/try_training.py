from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='Test response of model on training data with some specific information')
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')
parser.add_argument('--limit', nargs=1, type=int, default=[None])
parser.add_argument('--no-threading', action='store_true', default=False, help='Turn off threading')

args = parser.parse_args()
model_file = args.model
limit = args.limit[0]
threading = not args.no_threading
#parser.add_argument('--contest', type=str, choices=('voc', 'uiuc', 'uiuc-multiscale'), default='voc', help='Contest to try on')

import numpy as np
import amitgroup as ag
import gv
import glob
import os

np.set_printoptions(precision=2, suppress=True)

neg_files = sorted(glob.glob(os.path.join(os.environ['UIUC_DIR'], 'TrainImages', 'neg-*.pgm')))[:limit]
pos_files = sorted(glob.glob(os.path.join(os.environ['UIUC_DIR'], 'TrainImages', 'pos-*.pgm')))[:limit]

detector = gv.Detector.load(model_file)

num_features = detector.descriptor.num_parts

kernels = detector.prepare_kernels(None)

spread_bkg = detector.fixed_spread_bkg

eps = detector.settings['min_probability']
spread_bkg = np.clip(spread_bkg, eps, 1 - eps)

indices = np.where(spread_bkg <= 0.05)[0]
not_indices = np.where(spread_bkg > 0.05)[0]

weights = np.log(kernels[0] / (1 - kernels[0]) * ((1 - spread_bkg) / spread_bkg))

print 'max/min/mean weights', weights.max(), weights.min(), weights.mean()

weights = np.clip(weights, -2, 2)

def analyze_files(files):
    llhs = np.zeros((len(files), num_features))
    for i, fn in enumerate(files):
        im = gv.img.load_image(fn)
        im = gv.img.asgray(im)

        psize = detector.settings['subsample_size']

        rawfeats = detector.descriptor.extract_features(im)
        feats = gv.sub.subsample(rawfeats, psize)

        llh = np.apply_over_axes(np.sum, weights * feats, [0, 1]).ravel()
        llhs[i] = llh
    return llhs

print "Analyzing positive files"
llhs_pos = analyze_files(pos_files)

print "Analyzing negative files"
llhs_neg = analyze_files(neg_files)

print llhs_pos.shape
print llhs_pos.mean(), llhs_neg.mean()

feat_usefulness = llhs_pos.mean(axis=0) - llhs_neg.mean(axis=0)
#import pdb; pdb.set_trace()
print feat_usefulness
print feat_usefulness.mean()
print 'bottoms', len(indices), feat_usefulness[indices].mean()
print 'rest', len(not_indices), feat_usefulness[not_indices].mean()
print np.corrcoef(feat_usefulness, spread_bkg)[0,1]
#for i in xrange(num_features):
    #print llhs_pos[...,i].mean(), llhs_neg[...,i].mean()
