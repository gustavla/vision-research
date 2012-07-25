
from __future__ import print_function
from __future__ import division
import amitgroup as ag
import numpy as np
import matplotlib.pylab as plt
import sys

QUICK = '--quick' in sys.argv

num_mixtures = 9 

bedges_k = 4
inflate = True 

mixtures = []
if 1:
    path = "/var/tmp/local"
    
else:
    for d in range(10):
        digits, _ = ag.io.load_mnist('training', [d])
        if QUICK:
            digits = digits[:500]
        edges = ag.features.bedges(digits, inflate=inflate, k=bedges_k)
       
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

    return min_which

testing_digits, testing_labels = ag.io.load_mnist('testing')
if QUICK:
    testing_digits = testing_digits[:10]
testing_edges = ag.features.bedges(testing_digits, inflate=inflate, k=bedges_k)

N = len(testing_edges)
c = 0
for i, features in enumerate(testing_edges):
    label, comp = classify(features, mixtures)
    correct = label == testing_labels[i]
    c += int(correct)
    print(correct)
    if not correct:
        print("Label = {0}, component = {1}".format(label, comp))
        ag.plot.images(testing_digits[i])
        plt.show()
        ag.plot.images(np.rollaxis(features, 2))
        plt.show()
        ag.plot.images(np.rollaxis(mixtures[label].templates[comp], 2))
        plt.show()

print("Success rate: {0:.2f} ({1}/{2})".format(100*c/N, c, N))
