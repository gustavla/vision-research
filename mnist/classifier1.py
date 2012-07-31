
from __future__ import print_function
from __future__ import division
import amitgroup as ag
import numpy as np
import matplotlib.pylab as plt
import sys

QUICK = '--quick' in sys.argv
if QUICK:
    del sys.argv[sys.argv.index('--quick')]

bedges_k = 5 
inflate = True 
eps = 1e-3

        
# Classifer 
def classify(features, all_templates):
    # min loglikelihood
    min_cost = None
    min_which = None
    costs = []
    for digit, templates in enumerate(all_templates):
        # Clip them, to avoid 0 probabilities

        for mix_component, template in enumerate(templates):
            #assert features.shape == template.shape
            # Compare mixture with features
            cost = -np.sum(features * np.log(template) + (1-features) * np.log(1-template))
            #print("Cost {0} digit {1} comp {2}".format(cost, digit, mix_component))
            costs.append( (cost, digit, mix_component) )
            if min_cost is None or cost < min_cost:
                min_cost = cost 
                min_which = (digit, mix_component) 

    costs.sort()
    for cost in costs[::-1]:
        print(cost)

    return min_which


def main():
    try:
        mixtures_filename = sys.argv[1]
    except IndexError:
        raise ValueError("Usage: <mixtures file> [<inspect test index>]")

    try:
        inspect_component = int(sys.argv[2])
    except IndexError:
        inspect_component = None

    all_templates = np.load(mixtures_filename)['templates'] 


    if inspect_component is not None:
        #testing_digits, testing_labels = ag.io.load_mnist('testing', indices=slice(None, 10))
        #ag.io.load_mnist('testing', indices=inspect_component)
        digits, labels = ag.io.load_mnist('testing')
        digit, correct_label = digits[inspect_component], labels[inspect_component]
        
        features = ag.features.bedges(digit, inflate=inflate, k=bedges_k)

        label, comp = classify(features, all_templates)

        print("Digit: {0}".format(correct_label))
        print("Classified as: {0}".format(label))
        print(comp)
    else:
        testing_digits, testing_labels = ag.io.load_mnist('testing')
        if QUICK:
            testing_digits = testing_digits[:100]
        testing_edges = ag.features.bedges(testing_digits, inflate=inflate, k=bedges_k)

        N = len(testing_edges)
        c = 0
        all_templates = np.clip(all_templates, eps, 1.0 - eps)
        for i, features in enumerate(testing_edges):
            label, comp = classify(features, all_templates)
            correct = label == testing_labels[i]
            c += int(correct)
            print(i, N, correct)
            if False and not correct:
                print("Label = {0}, component = {1}".format(label, comp))
                ag.plot.images(testing_digits[i])
                plt.show()
                ag.plot.images(np.rollaxis(features, 2))
                plt.show()
                ag.plot.images(np.rollaxis(all_templates[label,comp], 2))
                plt.show()

        print("Success rate: {0:.2f} ({1}/{2})".format(100*c/N, c, N))

if __name__ == '__main__':
    #import cProfile as profile
    #profile.run('main()') 
    main()
