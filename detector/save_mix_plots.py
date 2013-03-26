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

# Load detector
detectors = []
for model_file in model_files:
    detector = gv.Detector.load(model_file)
    detectors.append(detector)

num_detectors = len(detectors)
num_parts = detector.descriptor.num_parts
num_mixtures = detector.num_mixtures

# Prepare kernels
kerns = []
bkgs = []
for detector in detectors:
    psize = detector.settings['subsample_size']
    radii = detector.settings['spread_radii']

    # Assumes fixed background model
    bkg = detector.background_model(np.ones((1, 1, 1)))
    sub_kernels = detector.prepare_kernels(bkg, settings=dict(spread_radii=radii, subsample_size=psize))
        
    bkgs.append(bkg)
    kerns.append(sub_kernels)

for p in xrange(num_parts):
    data = None
    if detector.support is None or 1:
        # Visualize feature activity if the support does not exist
        #assert 0, "This is broken since refactoring"
        data = detector.kernel_templates.sum(axis=-1)# / detector.kernel_templates.shape[-1] 
    else:
        data = detector.support

    sp = (num_detectors+1, num_mixtures)

    plt.clf()
    plt.subplot(sp[0], sp[1], 1)
    plt.imshow(detector.descriptor.visparts[p], interpolation='nearest', cmap=plt.cm.gray)

    sh = tuple([min([d.kernel_templates.shape[index] for d in detectors]) for index in (1,2)])

    k = 0
    for i in xrange(num_detectors):
        for j in xrange(num_mixtures):
            plt.subplot(sp[0], sp[1], num_mixtures+1+k).set_axis_off()
            #X = detectors[i].kernel_templates[j,...,p]
            Y = kerns[i][j,...,p]
            if 1:
                X = Y
            else:
                bkg = bkgs[i]
                print Y.min(), Y.max()
                print bkg
                X = np.log(Y / (1 - Y) * ((1 - bkg)/bkg))
        
            #if X.shape != sh:
                #diff = [X.shape[index] - sh[index] for index in (0, 1)]
                #X = X[diff[0]//2:diff[0]//2 + sh[0], diff[1]//2:diff[1]//2 + sh[1]]

            plt.imshow(X, interpolation='nearest', vmin=0, vmax=1)
            if captions:
                caption = captions[i]
            else:
                caption = detectors[i].settings['file']

            if num_mixtures > 1:
                caption += " mix:{0}".format(j)
            plt.title(caption)
            #plt.colorbar()
            k += 1

    plt.savefig('kernel-plots/parts-{0:03}.png'.format(p))
        
    #ag.plot.images(data, zero_to_one=True, caption=lambda i, im: "{0}: max: {1:.02} (w: {2:.02})".format(i, im.max(), detector.mixture.weights[i]))
