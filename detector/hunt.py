from __future__ import division
import matplotlib
matplotlib.use('Agg')

import argparse

parser = argparse.ArgumentParser(description='View mixture components')
parser.add_argument('models', metavar='<model file>', nargs='+', type=argparse.FileType('rb'), help='Filename of model files')
parser.add_argument('--captions', metavar='<caption>', nargs='*', type=str, help='Captions')

args = parser.parse_args()
model_files = args.models
captions = args.captions
if captions:
    assert len(captions) == len(model_files), "Must supply caption for all"


import matplotlib.pylab as plt
import amitgroup as ag
import gv
import numpy as np
import scipy.signal

# Load detector
detectors = []
for model_file in model_files:
    detector = gv.Detector.load(model_file)
    detectors.append(detector)

num_detectors = len(detectors)
num_parts = detector.descriptor.num_parts
num_mixtures = detector.num_mixtures

imbig = gv.img.load_image('/var/tmp/matlab/CarData/TestImages/test-0.pgm')
center = (48+20, 26+50)

im = imbig[center[0]-30:center[0]+30, center[1]-60:center[1]+60]
center = (30, 60)
#im[48+20, 26+50] = im.max() 

# Prepare kernels
kerns = []
bkgs = []
weights = []
feats = []
for detector in detectors:
    psize = detector.settings['subsample_size']
    radii = detector.settings['spread_radii']

    # Assumes fixed background model
    bkg = detector.background_model(np.ones((1, 1, 1)))
    sub_kernels = detector.prepare_kernels(bkg, settings=dict(spread_radii=radii, subsample_size=psize))
        
    bkgs.append(bkg)
    kerns.append(sub_kernels)
    Y = sub_kernels
    weights.append(np.log(Y / (1 - Y) * ((1 - bkg)/bkg)))

    feat = detector.descriptor.extract_features(im, settings=dict(spread_radii=radii, subsample_size=psize))
    feats.append(feat)

part_size = detectors[0].descriptor.settings['part_size']
adjusted_center = tuple([center[i] - part_size[i] + 1 for i in (0, 1)])

#import sys; sys.exit(0)

values = np.zeros((num_detectors, num_parts))

for p in xrange(num_parts):

    plt.clf()
    plt.figure(figsize=(12, 7))
    sh = (1, num_detectors+1)
    plt.subplot(sh[0], sh[1], 1)
    plt.imshow(im, cmap=plt.cm.gray)

    for i in xrange(num_detectors):
        #print weights[i][0,...,p].shape, feats[i].shape
        res = scipy.signal.convolve2d(feats[i][...,p], weights[i][0,...,p])
        plt.subplot(sh[0], sh[1], 2+i)
        plt.imshow(res, interpolation='nearest', vmin=-200, vmax=200, cmap=plt.cm.RdBu_r)
        #print res.shape
        #print adjusted_center
        adjusted_center = tuple([res.shape[k]//2 for k in range(2)])
        value = res[adjusted_center]
        values[i,p] = value

        #import pdb; pdb.set_trace()
        
    
        caption = captions[i] if captions is not None else ""
        caption += " {0:.2f}".format(value)
        #if captions is not None:
        plt.title(caption)
        #plt.colorbar()
    
    path = 'huntplots/hunt-{0:03}.png'.format(p)
    plt.savefig(path)
    print "Saved ", path

np.save('values.npy', values)
