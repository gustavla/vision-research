
from __future__ import division
import argparse
import sys

parser = argparse.ArgumentParser(description='Train coefficients')
parser.add_argument('features', metavar='<training features file>', type=argparse.FileType('rb'), help='Filename of features')
parser.add_argument('mixtures', metavar='<mixtures file>', type=argparse.FileType('rb'), help='Filename of mixtures')
parser.add_argument('output', metavar='<output file>', type=argparse.FileType('wb'), help='Filename of output')
parser.add_argument('-p', '--plot', action='store_true', help='Plot in real-time using pygame')
parser.add_argument('-i', dest='inspect', default=None, metavar='INDEX', type=int, help='Inspect a single element')
parser.add_argument('-d', '--deform', dest='deform', default='edges', type=str, choices=('edges', 'intensity'), help='What kind of features to perform the deformations on.')
parser.add_argument('--digit', dest='digit', nargs=1, metavar='DIGIT', type=int, help='Process only one digit')
parser.add_argument('-r', dest='range', nargs=2, metavar=('FROM', 'TO'), type=int, default=(0, sys.maxint), help='Range of frames, FROM (incl) and TO (excl)')
parser.add_argument('-l', dest='eta', nargs=1, default=[100000.0], type=float, help='Penalty coefficient eta')
parser.add_argument('--rho', dest='rho', nargs=1, default=[2.0], type=float, help='Penalty exponent rho')
parser.add_argument('-n', dest='iterations', nargs=1, default=[1], type=int, help='Number of iterations (reduces impact on initial eta/rho)')
parser.add_argument('-b', metavar='B', nargs=1, default=[0.0], type=float, help='Prior hypercoefficient b of Gamma distribution')

args = parser.parse_args()
features_file = args.features
mixtures_file = args.mixtures
output_file = args.output
PLOT = args.plot
digits = args.digit
inspect = args.inspect
deform_type = args.deform
print deform_type
eta = args.eta[0]
rho = args.rho[0]
b0 = args.b[0]
n0, n1 = args.range
ITERS = args.iterations[0]

import amitgroup as ag
import numpy as np
import time
from copy import copy
from classifier import add_prior

features_data = np.load(features_file)
mixtures_data = np.load(mixtures_file)
all_templates = mixtures_data['templates']
all_affinities = mixtures_data['affinities']

if deform_type == 'intensity':
    try:
        all_graylevels = features_data['originals']
        all_graylevel_templates = mixtures_data['graylevel_templates']
    except KeyError:
            raise Exception("The feature file must be run with --save-originals, and the mixtures must be trained with this file")

mixtures_meta = mixtures_data['meta'].flat[0]
M = mixtures_meta['mixtures'] 

if digits is not None:
    shape = (1, M)
    d0 = digits[0]
else:
    digits = range(10)
    shape = (10, M)
    d0 = 0

level_capacity = 3

im_shape = mixtures_meta['shape']
sh = (ITERS + 1,) + shape + ag.util.DisplacementFieldWavelet.shape_for_size(im_shape, level_capacity=level_capacity)

# Storage for means, variances and how many samples were used to calculate these 
means = np.empty(sh)
variances = np.empty(sh)
llh_variances = np.empty((ITERS + 1,) + shape + im_shape)
samples = np.empty(sh)

#llh_sh = shape 
#llh_means = np.empty(llh_sh)
#llh_variances = np.empty(llh_sh)
all_digit_features = features_data['features'] 

totcost = 0.0

meta = {}


for loop in xrange(1, ITERS + 1):
    for d in digits:
        entries = [[] for i in xrange(M)]
        slices = [[] for i in xrange(M)]
        all_features = all_digit_features[d]
        n1 = min(n1, len(all_features))
        if inspect is not None:
            n0 = inspect
            n1 = n0+1 

        us = []
        for i in xrange(n0, n1):
            affinities = all_affinities[d,i]
            m = np.argmax(affinities)
            #F = np.rollaxis(all_templates[d,m], axis=2)
            #I = np.rollaxis(all_features[i], axis=2).astype(float)

            settings = dict(    
                tol=0.1, 
                maxiter=200, 
                start_level=2, 
                last_level=level_capacity,
                wavelet='db4'
            )
            
            if loop == 1:
                settings['penalty'] = eta 
                settings['rho'] = rho

                # Save these initial values
                samples[0, d-d0, m] = 0 # not applicable 
                means[0, d-d0, m] = 0.0 
                variances[0, d-d0, m] = 1/ag.util.DisplacementFieldWavelet.make_lambdas(im_shape, level_capacity, eta=eta, rho=rho)

                # Save the initial settings as part of meta
                meta = copy(settings)
                meta['b0'] = b0
        
                # Leave out llh_variances (same as setting it to 1.0, initially)
            else:
                settings['means'] = means[loop-1,d-d0,m]
                settings['variances'] = variances[loop-1,d-d0,m]
                if deform_type == 'intensity':
                    settings['llh_variances'] = llh_variances[loop-1,d-d0,m]
        
          
            t1 = time.time()
            if deform_type == 'edges':
                F = all_templates[d,m]
                I = all_features[i].astype(float)

                imdef, info = ag.stats.bernoulli_deformation(F, I, debug_plot=PLOT, **settings)

            elif deform_type == 'intensity':
                F = all_graylevel_templates[d,m]
                I = all_graylevels[d,i]

                settings['tol'] = 0.000001
                settings['maxiter'] = 50

                imdef, info = ag.stats.image_deformation(F, I, debug_plot=PLOT, **settings) 

                slices[m].append(imdef.deform(F))
                # Save slices
                #Fdef = np.asarray([
                #    imdef.deform(F[j]) for j in xrange(8)
                #])
                #slices[m].append(Fdef - I)
            
            else:
                raise Exception("Invalid deform type selection")
            t2 = time.time()

            if imdef is None:
                sys.exit(0) 

            print "{5}/{6} {3:.02f}% Digit: {0} Index: {1} (time = {2} s) min cost: {4}".format(d, i, t2-t1, 100*(d+(1+i-n0)/(n1-n0))/10, info['cost'], loop, ITERS)
            totcost += info['cost']
             
            entries[m].append(imdef.u)
        
        for m in xrange(M):
            data = np.asarray(entries[m])
            assert len(data) >= 2 #, "Need more data! (some mixture components had not a single data point" 

            #print means.shape, data.shape
            # Prior
            #print d, m, means.shape
            samples[loop, d-d0, m] = data.shape[0]
            means[loop, d-d0, m] = data.mean(axis=0) 
            variances[loop, d-d0, m] = data.var(axis=0) 

            if deform_type == 'intensity':
                # Include variance of likelihood int he variance of the prior.
                #variances[loop, d-d0, m]

                # Variance of likelihood 
                slicesm = np.asarray(slices[m])
                llh_var = slicesm.var(axis=0)
                if 0:
                    import pylab as plt
                    plt.imshow(llh_var)
                    plt.colorbar()
                    plt.show()
                    import sys; sys.exit(0)
                #variances[loop, d-d0, m] /= llh_var.mean()
                llh_variances[loop, d-d0, m] = np.clip(llh_var, 0.05, np.inf)

                # Make sure llh_variances is normalized
                llhm = llh_variances.mean()

                # Shift over nominal inflation value from likelihood to prior.
                #llh_variances[loop, d-d0, m] /= llhm
                #variances[loop, d-d0, m] *= llhm 
        
                #variancesc
            else:
                # Smooth it with a prior
                if eta is not None and rho is not None and b0 > 0: 
                    variances[loop, d-d0, m] = add_prior(variances[loop, d-d0, m], level_capacity, im_shape, eta, rho, b0, samples[loop,d-d0,m])

            #print len(data)
            #print variances[loop,d-d0,m].min(), variances[loop,d-d0,m].max()
            #ag.plot.images(variances[loop,d-d0,m], zero_to_one=False)
            #import sys; sys.exit(0)

            # Likelihood
            #values = np.asarray(slices[m]).flatten()
            #np.save("tmp-values.{0}.{1}.npy".format(d, m), values)
            #np.save("tmp-values.{0}.{1}.npy".format(d, m), data)

            #llh_means[d-d0, m] = values.mean()
            #llh_variances[d-d0, m] = values.var()

additional = {}
if ITERS > 1:
    additional['all_iterations_mean'] = means
    additional['all_iterations_var'] = variances
    additional['all_iterations_samples'] = samples # TODO: Does this change?

if deform_type == 'intensity':
    if ITERS > 1:
        additional['all_llh_var'] = llh_variances
    additional['llh_var'] = llh_variances[-1]
             
np.savez(output_file, prior_mean=means[-1], prior_var=variances[-1], samples=samples[-1], meta=meta, **additional)
