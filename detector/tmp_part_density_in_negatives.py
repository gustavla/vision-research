from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='Train mixture model on edge data')
parser.add_argument('settings', metavar='<settings file>', type=argparse.FileType('r'), help='Filename of settings file')
parser.add_argument('--limit', type=int, default=10)

args = parser.parse_args()
settings_file = args.settings
limit = args.limit

import gv
import os
import os.path
import glob
import numpy as np
import amitgroup as ag
import itertools
from scipy.stats import mstats
from train_superimposed import generate_random_patches
from settings import load_settings

settings = load_settings(settings_file)
descriptor = gv.load_descriptor(settings)

path = os.path.expandvars(settings['detector']['neg_dir'])
files = sorted(glob.glob(path))

gen = generate_random_patches(files, (100, 100), seed=0, per_image=5)

cut = 4
radii = settings['detector']['spread_radii']
psize = settings['detector']['subsample_size']

mm = []

for i, im in itertools.islice(enumerate(gen), limit):
    print 'Processing {}'.format(i)
    feats = descriptor.extract_features(im, settings=dict(spread_radii=radii, subsample_size=psize, crop_border=cut))

    m = feats.mean()
    mm.append(m)

import matplotlib.pylab as plt
plt.hist(mm, 100)
plt.show()
