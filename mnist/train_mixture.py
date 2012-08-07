from __future__ import print_function
import amitgroup as ag
from amitgroup.stats import BernoulliMixture
import numpy as np
import sys


try:
    filename = sys.argv[1]
    output_filename = sys.argv[2]
    k = int(sys.argv[3])
    #inflate = 'inflate' in sys.argv
except IndexError:
    print("<intput filename> <output filename> <num mix components>")
    sys.exit(0)

data = np.load(filename)

def train_mixture(data):
    all_templates = []
    all_weights = []
    all_affinities = []
    for d in range(10):
        print(d)
        str_d = str(d)
        digits = data[str_d]
        #digits = digits[:50]

        # Train a mixture model
        mixture = BernoulliMixture(k, digits)
        mixture.run_EM(1e-3, min_probability=0.01)
        
        #mixture.save('mix/mixture-digit-{0}'.format(d))
        all_templates.append(mixture.templates)
        all_weights.append(mixture.weights)
        all_affinities.append(mixture.affinities)

    np.savez(output_filename, templates=all_templates, weights=all_weights, affinities=mixture.affinities) 

if __name__ == '__main__':
    #import cProfile as profile
    #profile.run('train_mixture(data)')
    train_mixture(data)
