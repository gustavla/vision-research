from __future__ import division, print_function
import argparse

parser = argparse.ArgumentParser(description='Test response of model')
parser.add_argument('models', metavar='<model file>', nargs='+', type=argparse.FileType('rb'), help='Filename of model files')
parser.add_argument('--captions', metavar='<caption>', nargs='*', type=str, help='Captions')

args = parser.parse_args()
model_files = args.models
captions = args.captions
if captions:
    assert len(captions) == len(model_files), "Must supply caption for all"

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import scipy.integrate
import glob
import os
import gv

# Load detector
detectors = []
for model_file in model_files:
    detector = gv.Detector.load(model_file)
    detectors.append(detector)

# Prepare kernels
kerns = []
bkgs = []
weights = []
for detector in detectors:
    psize = detector.settings['subsample_size']
    radii = detector.settings['spread_radii']

    # Assumes fixed background model
    bkg, bkg_test = detector.background_model(np.ones((1, 1, 1)))
    sub_kernels = detector.prepare_kernels(bkg, settings=dict(spread_radii=radii, subsample_size=psize))
        
    bkgs.append(bkg)
    kerns.append(sub_kernels)

    Y = sub_kernels
    weights.append(np.log(Y / (1 - Y) * ((1 - bkg)/bkg)))

num_detectors = len(detectors)
num_parts = detector.descriptor.num_parts
num_mixtures = detector.num_mixtures

test_files = sorted(glob.glob('/var/tmp/matlab/uiuc-test/*.png'))

tot = [None for i in xrange(num_detectors)]

assert test_files, "No test files"

for f in test_files:
    print("Processing file", f)

    psize = detectors[0].settings['subsample_size']
    radii = detectors[0].settings['spread_radii']
    
    # They share descriptor (hopefully), so let's use that
    im = gv.img.load_image(f)
    feats_big = detectors[0].descriptor.extract_features(im, settings=dict(spread_radii=radii, subsample_size=psize))         
    feats = gv.sub.subsample(feats_big, psize)
    
    #print('feats', feats.shape)

    for i in xrange(num_detectors):
        #print('w', weights[i][0].shape, feats.shape)
        res = weights[i][0] * feats
    
        if tot[i] is None:
            tot[i] = np.zeros_like(res)

        tot[i] += res

for i in xrange(num_detectors):
    tot[i] /= len(test_files)
# Iterate through parts and save images
         
for p in xrange(num_parts):
    plt.clf()
    sp = (num_detectors, 1)
    for i in xrange(num_detectors):
        plt.subplot(sp[0], sp[1], 1+i).set_axis_off()
        plt.imshow(tot[i][...,p], interpolation='nearest', cmap=plt.cm.RdBu_r, vmin=-2, vmax=2) 
        #plt.title('part-{0}'.format(p))
        caption = captions[i] if captions is not None else ""
        caption += " {0:.2f}".format(tot[i][...,p].mean())
        plt.title(caption)
    plt.savefig('check-plots/part-{0}.png'.format(p))

# Do one with all of them together
plt.clf()
for i in xrange(num_detectors):
    plt.subplot(sp[0], sp[1], 1+i).set_axis_off()
    plt.imshow(tot[i].mean(axis=-1), interpolation='nearest', cmap=plt.cm.RdBu_r, vmin=-0.1, vmax=0.1) 
    caption = captions[i] if captions is not None else ""
    caption += " {0:.2f}".format(tot[i].mean())
    plt.title(caption)
plt.savefig('check-plots/all.png')
