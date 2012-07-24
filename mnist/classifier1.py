
from __future__ import print_function
from __future__ import division
import amitgroup as ag
import numpy as np
import sys

QUICK = '--quick' in sys.argv

num_mixtures = 4

mixtures = []
for d in range(10):
    digits, _ = ag.io.load_mnist('training', [d])
    if QUICK:
        digits = digits[:2000]
    edges = ag.features.bedges(digits)
   
    #edges_flat = edges.flatten()
    #edges_1d = np.array([edges_flat])

    # Train the mixture model
    print("Training mixture model for digit", d)
    mixture = ag.stats.BernoulliMixture(num_mixtures, edges)
    mixture.run_EM(1e-4, save_template=True)
    
    mixtures.append(mixture) 

# Classifer 
def classify(features, mixtures):
    eps = 1e-3
    lookup = []
    # min loglikelihood
    min_cost = None
    min_which = None
    for digit, mixture in enumerate(mixtures):
        templates = mixture.get_templates()
        # Clip them, to avoid 0 probabilities
        templates = np.clip(templates, eps, 1.0 - eps)

        for mix_component, template in enumerate(templates):
            assert features.shape == template.shape
            # Compare mixture with features
            cost = -np.sum(features * np.log(templates) + (1-features) * np.log(1-templates))
            if min_cost is None or cost < min_cost:
                min_cost = cost 
                min_which = (digit, mix_component) 

    return min_which[0]

testing_digits, testing_labels = ag.io.load_mnist('testing')
if QUICK:
    testing_digits = testing_digits[:50]
testing_edges = ag.features.bedges(testing_digits)

N = len(testing_edges)
c = 0
for i, features in enumerate(testing_edges):
    label = classify(features, mixtures)
    correct = label == testing_labels[i]
    c += int(correct)

print("Success rate: {0:2f} ({1}/{2})".format(100*c/N, c, N))
