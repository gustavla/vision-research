from __future__ import print_function
from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='Test response of model')
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')
#parser.add_argument('mixcomp', metavar='MIXCOMP', type=int, help='Mixture component')

args = parser.parse_args()
model_file = args.model
#mixcomp = args.mixcomp

import gv
import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import sys
import amitgroup as ag
import glob
import os

ag.set_verbose(True)

detector = gv.Detector.load(model_file)

files = glob.glob(os.path.join(detector.settings['cad_dir'], "*.png"))

np.random.seed(2)
#bkg = np.random.uniform(0.05, 0.80, detector.num_features)

B = 100
#bkgs = [np.random.uniform(0.05, 0.60, detector.num_features) for i in xrange(B)]
bkg = np.random.uniform(0.05, 0.40, detector.num_features)
bkgs = [bkg]

# Preprocess kernel
sub_kernels_all = [detector.prepare_kernels(bkg) for bkg in bkgs]


#llhs = [[[] for j in xrange(B)] for i in xrange(detector.num_mixtures)]
llhs = [[] for i in xrange(detector.num_mixtures)]
for i, edges in enumerate(detector.gen_img(files, actual=True)):
    # Which mixture?
    comp = np.argmax(detector.mixture.affinities[i])
    ag.info('Processing', i) 

    # Check log likelihood of this against the model
    # and add to list
    #llh = np.sum(edges * np.log(kern) + (1-edges) * np.log(1-kern))

    for j, bkg in enumerate(bkgs):
        kern = sub_kernels_all[j][comp]
        a = np.log(kern/(1-kern) * ((1-bkg)/bkg))
        llh = np.sum((edges-bkg) * a)
        summand = a**2 * bkg * (1-bkg) 
        llh /= np.sqrt(summand.sum())
        #llhs[comp][j].append(llh)
        llhs[comp].append(llh)


mn = 0 
mx = 100 
dt = 1.0 
bins = np.arange(mn, mx+1e-10, dt)


if 1:
    gs = gridspec.GridSpec(detector.num_mixtures, 7)
    
    for i, llh in enumerate(llhs):
        ax = plt.subplot(gs[detector.num_mixtures-1-i,:-1])
        #plt.subplot(detector.num_mixtures, 1, 1+i)
        plt.hist(llh, bins=bins, alpha=1, normed=True, label='Component {0}'.format(i))

        ax2 = plt.subplot(gs[detector.num_mixtures-1-i,-1]).set_axis_off()
        plt.imshow(detector.support[i], cmap=plt.cm.gray)

#plt.legend()
#import pdb; pdb.set_trace()
    plt.show()
else:
    means = np.empty((detector.num_mixtures, B))
    for i in xrange(detector.num_mixtures):
        for j in xrange(B):
            means[i,j] = np.mean(llhs[i][j])

    try:
        supsize = detctor.support.sum(axis=0)
    except:
        import pdb; pdb.set_trace()
    np.savez('stuff.npz', means=means, supsize=supsize) 
