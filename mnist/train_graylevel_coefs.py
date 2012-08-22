
from __future__ import division
import argparse
import sys

parser = argparse.ArgumentParser(description='Train coefficients for graylevel deformations')
parser.add_argument('features', metavar='<training features file>', type=argparse.FileType('rb'), help='Filename of features')
parser.add_argument('mixtures', metavar='<mixtures file>', type=argparse.FileType('rb'), help='Filename of mixtures')
parser.add_argument('output', metavar='<output file>', type=argparse.FileType('wb'), help='Filename of output')
parser.add_argument('-p', '--plot', action='store_true', help='Plot in real-time using pygame')
parser.add_argument('-i', dest='inspect', nargs=1, default=[None], metavar='INDEX', type=int, help='Inspect a single element')
parser.add_argument('-d', dest='digit', nargs=1, metavar='DIGIT', type=int, help='Process only one digit')
parser.add_argument('-r', dest='range', nargs=2, metavar=('FROM', 'TO'), type=int, default=(0, sys.maxint), help='Range of frames, FROM (incl) and TO (excl)')
parser.add_argument('-l', dest='penalty', nargs=1, default=[100000.0], type=float, help='Penalty term')
parser.add_argument('--rho', dest='rho', nargs=1, default=[2.0], type=float, help='Penalty exponent rho')

args = parser.parse_args()
features_file = args.features
mixtures_file = args.mixtures
output_file = args.output
PLOT = args.plot
digits = args.digit
inspect = args.inspect[0]
penalty = args.penalty[0]
n0, n1 = args.range
rho = args.rho[0]

import amitgroup as ag
import numpy as np
import time

features_data = np.load(features_file)
mixtures_data = np.load(mixtures_file)
try:
    all_originals = features_data['originals']
    all_templates = mixtures_data['graylevel_templates']
except KeyError:
    raise Exception("The feature file must be run with --save-originals")
all_affinities = mixtures_data['affinities']

meta = mixtures_data['meta'].flat[0]
M = meta['mixtures'] 

if digits is not None:
    shape = (1, M)
    d0 = digits[0]
else:
    digits = range(10)
    shape = (10, M)
    d0 = 0

im_shape = meta['shape']
sh = shape + ag.util.DisplacementFieldWavelet.shape_for_size(im_shape, level_capacity=3)

# Storage for means, variances and how many samples were used to calculate these 
means = np.empty(sh)
variances = np.empty(sh)
samples = np.empty(sh)

#llh_sh = shape 
#llh_means = np.empty(llh_sh)
#llh_variances = np.empty(llh_sh)
all_digit_features = features_data['features'] 

for d in digits:
    entries = [[] for i in range(M)]
    slices = [[] for i in range(M)]
    all_features = all_digit_features[d]
    n1 = min(n1, len(all_features))
    if inspect is not None:
        n0 = inspect
        n1 = n0+1 

    us = []
    for i in range(n0, n1):
        affinities = all_affinities[d,i]
        m = np.argmax(affinities)
        #F = np.rollaxis(all_templates[d,m], axis=2)
        #I = np.rollaxis(all_features[i], axis=2).astype(float)
        F = all_templates[d,m]
        #I = all_features[i].astype(float)
        I = all_originals[d,i]
        print F.shape, I.shape

        x, y = ag.util.DisplacementFieldWavelet.meshgrid_for_shape(F.shape)

        settings = dict(    
            penalty=penalty, 
            rho=rho, 
            tol=0.00001, 
            maxiter=100, 
            start_level=1, 
            last_level=3, 
            wavelet='db4'
        )

        t1 = time.time()
        #imdef, info = ag.stats.bernoulli_deformation(F, I, debug_plot=PLOT, **settings)
        imdef, info = ag.stats.image_deformation(F, I, debug_plot=PLOT, **settings)
        t2 = time.time()
        print "{3:.02f}% Digit: {0} Index: {1} (time = {2} s)".format(d, i, t2-t1, 100*(d+(1+i-n0)/(n1-n0))/10)

        imdef or sys.exit(0)

        entries[m].append(imdef.u)
        Fdef = imdef.deform(F)
        slices[m].append(Fdef - I)

    for m in range(M):
        data = np.asarray(entries[m])
        assert len(data) > 0, "Need more data!" 

        print means.shape, data.shape
        # Prior
        print d, m, means.shape

        llhs = np.asarray(slices[m])

        samples[d-d0, m] = data.shape[0]
        means[d-d0, m] = data.mean(axis=0)
        variances[d-d0, m] = data.var(axis=0) / llhs.var()

        # Likelihood
        #np.save("tmp-values.{0}.{1}.npy".format(d, m), values)
        #np.save("tmp-values.{0}.{1}.npy".format(d, m), data)

        #llh_means[d-d0, m] = values.mean()
        #llh_variances[d-d0, m] = values.var()
         
np.savez(output_file, prior_mean=means, prior_var=variances, samples=samples, meta=settings)
