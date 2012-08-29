from __future__ import print_function
import argparse

eps_default = 0.05

parser = argparse.ArgumentParser(description='Train mixture model on edge data')
parser.add_argument('features', metavar='<features file>', type=argparse.FileType('rb'), help="Filename of feature file (npz)")
parser.add_argument('output', metavar='<output file>', type=argparse.FileType('wb'), help='Filename of output file')
parser.add_argument('mixtures', metavar='MIXTURES', type=int, help='Number of mixture components')
parser.add_argument('-e', dest='epsilon', nargs=1, default=[eps_default], metavar='EPSILON', type=float, help='Minimum probability of a mixture components, as well as (1-EPSILON) being the maximum probability. Defaults to {0}.'.format(eps_default))
parser.add_argument('-s', dest='seed', nargs=1, default=[0], metavar='SEED', type=int, help='Mixture model seed')
parser.add_argument('-p', '--plot', action='store_true', help='Plot in real-time using pygame')

args = parser.parse_args()
feat_file = args.features
output_file = args.output
M = args.mixtures
eps = args.epsilon[0]
PLOT = args.plot
seed = args.seed[0]

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
    for d in xrange(10):
        print("Training digit", d)
        digits = features[d] 

        # Train a mixture model
        mixture = BernoulliMixture(M, digits, init_seed=seed)
        mixture.run_EM(1e-10, min_probability=eps, debug_plot=PLOT)
        
        #mixture.save('mix/mixture-digit-{0}'.format(d))
        all_templates.append(mixture.templates)
        all_weights.append(mixture.weights)
        all_affinities.append(mixture.affinities)

    all_templates = np.asarray(all_templates)
    all_weights = np.asarray(all_weights)
    all_affinities = np.asarray(all_affinities)

    save_dict = dict(
        templates=all_templates,
        weights=all_weights, 
        affinities=all_affinities,
        meta=dict(mixtures=M, eps=eps, shape=meta['shape'], seed=seed)
    )

    if 'originals' in data:
        all_originals = data['originals']
        all_graylevel_templates = np.empty((10, M, 32, 32))
        # Save graylevel templates as well
        for d in xrange(10):
            #for i in xrange(len(all_affinities[d])):
            for m in xrange(M):
                #np.rollaxis(all_affinities[d], 1) *     
                #all_graylevel_templates = (all_affinities[d,:,m] * all_originals).mean()
                all_graylevel_templates[d,m] = np.average(all_originals[d], axis=0, weights=all_affinities[d,:,m])

        save_dict['graylevel_templates'] = all_graylevel_templates

    np.savez(output_file, **save_dict)

if __name__ == '__main__':
    train_mixture(data)
