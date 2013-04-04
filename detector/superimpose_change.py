from __future__ import division
from settings import load_settings

import argparse

parser = argparse.ArgumentParser(description="Experimentation")
parser.add_argument('settings', metavar='<settings file>', type=argparse.FileType('r'), help='Filename of settings file')

args = parser.parse_args()
settings_file = args.settings
settings = load_settings(settings_file)

#import matplotlib
#matplotlib.use('Agg')
import glob
import sys
import os
import gv
import numpy as np
import amitgroup as ag
import matplotlib.pylab as plt

from superimpose_experiment import *
from operator import itemgetter

descriptor = gv.load_descriptor(settings)

def get_edges(settings, config):
    offset = settings['detector'].get('train_offset', 0)
    limit = settings['detector'].get('train_limit')
    if limit is not None:
        limit += offset
    files = sorted(glob.glob(settings['detector']['train_dir']))[offset:limit] * settings['detector'].get('duplicate', 1)
    alpha_and_images = map(load_and_crop, files)
    if alpha_and_images[0][0] is None:
        alpha = None
        all_alphas = None
    else:
        all_alphas = np.asarray(map(itemgetter(0), alpha_and_images))
        #all_alphas = np.asarray(map(lambda x: x[0], alpha_and_images))
        alpha = all_alphas[:,7-4:8+4,7-4:8+4].mean(axis=0)
    images = np.asarray(map(itemgetter(1), alpha_and_images))


    if config.startswith('bkg'):
        seed = int(config[3:])
        neg_gen = generate_random_patches(neg_filenames, (15, 15), seed=seed)
        for i in xrange(len(images)):
            # Superimpose it onto the negative patch
            images[i] = neg_gen.next()
        
    elif config.startswith('sup'):
        seed = int(config[3:])
        neg_gen = generate_random_patches(neg_filenames, (15, 15), seed=seed)
        for i in xrange(len(images)):
            # Superimpose it onto the negative patch
            images[i] = composite(images[i], neg_gen.next(), all_alphas[i])

    all_edges = ag.features.bedges(images, **settings['edges'])
    aedges = ag.features.bedges(all_alphas[0].astype(np.float64), k=3, radius=0, contrast_insensitive=True)[2:-2,2:-2,0]
    return all_edges[:,1:-1,1:-1].astype(np.bool), images, alpha.astype(np.bool), aedges

def printb(x):
    print x.astype(np.uint8)

edges0, images0, alpha, aedges = get_edges(settings, 'none')
edges1, images1 = get_edges(settings, 'bkg0')[:2]
edges2, images2 = get_edges(settings, 'sup0')[:2]

# Code the parts
coded_parts = np.asarray(map(lambda x: descriptor.extract_parts(x.astype(np.uint8))[0,0], edges2))

if 0: # Display background distribution
    plt.plot(coded_parts.mean(axis=0))
    plt.show()

if 0:
    f = 136 
    ee = edges2[coded_parts[:,f] == 1]
    pe = descriptor.parts[f]
    llh = ee * np.log(pe) + (1 - ee) * np.log(1 - pe) 
    avg = llh.mean(axis=0)
    plt.clf()
    print avg.min(), avg.max()
    for i, X in enumerate(np.rollaxis(avg, 2)):
        plt.subplot(2, 2, 1+i)
        plt.imshow(X, interpolation='nearest', vmin=-4, vmax=0, cmap=plt.cm.gray)
    plt.show()

eqv12 = ~(edges1 ^ edges2)
eqv02 = ~(edges0 ^ edges2)
either = eqv12 | eqv02

disappeared = (~eqv02 & edges0)
appeared = alpha.reshape(alpha.shape+(1,)) & (~edges0 & edges2)

dis = disappeared.mean(axis=0)
app = appeared.mean(axis=0)

# Create a model for each different background patch
num_features = coded_parts.shape[1]
mods = np.zeros((num_features, 2, 9, 9, 4)) # Don't hard code this
for f in xrange(num_features):
    parts = coded_parts[:,f] == 1
    if parts.sum() > 0:
        mods[f,0] = disappeared[parts].mean(axis=0)
        mods[f,1] = appeared[parts].mean(axis=0)
    
np.save('_mods.npy', mods)
    #coded_parts


def plot_edgemaps(images):
    ag.plot.images(np.rollaxis(images, 2))

#import ipdb; ipdb.set_trace()

combo = np.array([disappeared, appeared])
print 'combo shape', combo.shape
combo = np.rollaxis(combo, 0, start=2)
print 'combo shape', combo.shape
#combo = np.rollaxis(combo, 4, start=2)
#print 'combo shape', combo.shape

bm = ag.stats.BernoulliMixture(10, combo)
bm.run_EM(1e-8, 1e-10)

bm.save('_mix.npy')

#import pdb; pdb.set_trace()

np.save('_blackout.npy', dis)
np.save('_blackin.npy', app)

if 0:
    ag.plot.images(np.rollaxis(dis, 2))
    ag.plot.images(np.rollaxis(app, 2))

# Do a histogram
fl = disappeared.mean(axis=0).ravel()
if 0:
    plt.hist(fl[fl > 0])
    plt.show()
#import pdb; pdb.set_trace()

i = 1
printb(edges0[i,...,0])
print '-'
printb(edges1[i,...,0])
print '-'
printb(edges2[i,...,0])
print '-eq bkg-sup'
printb(eqv12[i,...,0])
print '-eq non-sup'
printb(eqv02[i,...,0])
print 'both'
printb(either[i,...,0])

print '-dis'
printb(disappeared[i,...,0])
print '-'

printb(alpha)

print disappeared[...,1].mean(axis=0)
#import pylab as plt
#ag.plot.images(np.rollaxis(disappeared.mean(axis=0), 2), zero_to_one=True)

print edges0.shape, edges1.shape
