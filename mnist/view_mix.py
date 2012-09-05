import argparse

parser = argparse.ArgumentParser(description='Extract MNIST features (zero-padded to 32 x 32) arranged for training (indexed by label)')
parser.add_argument('features', metavar='<features file>', type=argparse.FileType('rb'), help="Filename of feature file (npz)")
parser.add_argument('mixtures', metavar='<mixtures file>', type=argparse.FileType('rb'), help='Filename of mixtures')
parser.add_argument('digit', type=int, help='Digit')
parser.add_argument('component', type=int, help='Mixture component')

args = parser.parse_args()
feat_file = args.features
mix_file = args.mixtures
d = args.digit
m = args.component

import numpy as np
import amitgroup as ag

mixtures_data = np.load(mix_file)
features_data = np.load(feat_file)

originals = features_data['originals']
aff = mixtures_data['affinities']

indices = np.arange(aff.shape[1])[(aff[d,:,m] > 0.5)]
im = originals[d][indices]
ag.plot.images(list(im) + [im.mean(axis=0)])
