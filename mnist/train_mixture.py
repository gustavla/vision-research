from __future__ import print_function
import argparse

eps_default = 0.01

parser = argparse.ArgumentParser(description='Train mixture model on edge data')
parser.add_argument('features', metavar='<features file>', type=argparse.FileType('rb'), help="Filename of feature file (npz)")
parser.add_argument('output', metavar='<output file>', type=argparse.FileType('wb'), help='Filename of output file')
parser.add_argument('mixtures', metavar='MIXTURES', type=int, help='Number of mixture components')
parser.add_argument('-e', dest='epsilon', nargs=1, default=[eps_default], metavar='EPSILON', type=float, help='Minimum probability of a mixture components, as well as (1-EPSILON) being the maximum probability. Defaults to {0}.'.format(eps_default))

args = parser.parse_args()
feat_file = args.features
output_file = args.output
M = args.mixtures
eps = args.epsilon[0]

import amitgroup as ag
from amitgroup.stats import BernoulliMixture
import numpy as np
import sys

data = np.load(feat_file)
meta = data['meta'].flat[0]

def train_mixture(data):
    features = data['features']
    assert features.ndim == 5, "Sure you used extract_training_features.py to generate this?"
    all_templates = []
    all_weights = []
    all_affinities = []
    for d in range(10):
        print("Training digit", d)
        digits = features[d] 

        # Train a mixture model
        mixture = BernoulliMixture(M, digits)
        mixture.run_EM(1e-10, min_probability=eps)
        
        #mixture.save('mix/mixture-digit-{0}'.format(d))
        all_templates.append(mixture.templates)
        all_weights.append(mixture.weights)
        all_affinities.append(mixture.affinities)

    all_templates = np.asarray(all_templates)
    all_weights = np.asarray(all_weights)
    all_affinities = np.asarray(all_affinities)

    np.savez(output_file, 
             templates=all_templates, 
             weights=all_weights, 
             affinities=all_affinities,
             meta=dict(mixtures=M, eps=eps, shape=meta['shape']))

if __name__ == '__main__':
    train_mixture(data)
