
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
def classify(features, all_templates, means, variances, llh_means, llh_variances):
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

    costs = filter(lambda t: t[0] < min_cost * 1.1, costs)
    costs.sort()

    # If all the costs left are the same digit, don't bother doing the deformation
    checked = [] 
    for t in costs:
        if t[1] not in checked:
            checked.append(t[1])
    
    if len(checked) == 1:
        return min_which

    # Filter so that we have only one of each mixture
    if 0:
        for i, t in enumerate(costs):
            if t[1] not in checked:
                checked.append(t[1])
            else:
                del[costs[i]]
    for t in costs:
        cost, digit, mix_component = t 
        print("Doing cost {0} digit {1} mix_component {2}".format(*t))
        # Do a deformation
        means[digit, mix_component]
    
        F = np.rollaxis(all_templates[digit][mix_component], axis=2)
        I = np.rollaxis(features, axis=2)
    
        #means=means[digit, mix_component], variances=variances[digit, mix_component], 
        #variances[digit, mix_component, :, 0, :, :, :] = 0.0009765625 
        #variances[digit, mix_component, :, 1, :, :, :] = 0.00390625 
        #variances[digit, mix_component, :, 2, :, :, :] = 0.0625 
        #variances[digit, mix_component, :, 3, :, :, :] = 16.0 
        #variances[digit, mix_component, :, -2:] = 0.0
        llh_variances = np.clip(llh_variances, 0.0001, 10000000)
        var = variances[digit, mix_component]/llh_variances[digit, mix_component]

        imdef, info = ag.ml.bernoulli_deformation(F, I, wavelet='db8', stepsize_scale_factor=0.01, means=means[digit, mix_component], variances=var, last_level=3, debug_plot=True, tol=0.00004)
        # Update the cost 
        print(t, 'new cost:', info['costs'][-1])

    return min_which


def main():
    try:
        mixtures_filename = sys.argv[1]
        coef_filename = sys.argv[2]
    except IndexError:
        print("Usage: <mixtures file> <coef file> [<inspect test index>]")
        sys.exit(0)

    try:
        inspect_component = int(sys.argv[3])
    except IndexError:
        inspect_component = None

    all_templates = np.load(mixtures_filename)['templates'] 
    coefs = np.load(coef_filename)
    
    means = coefs['prior_mean']
    variances = coefs['prior_var']
    llh_means = coefs['llh_mean']
    llh_variances = coefs['llh_var']

    if inspect_component is not None:
        #testing_digits, testing_labels = ag.io.load_mnist('testing', indices=slice(None, 10))
        #ag.io.load_mnist('testing', indices=inspect_component)
        digits, labels = ag.io.load_mnist('testing')
        digits = ag.util.zeropad(digits, (0, 2, 2))
        digit, correct_label = digits[inspect_component], labels[inspect_component]
        
        features = ag.features.bedges(digit, inflate=inflate, k=bedges_k)

        label, comp = classify(features, all_templates, means, variances, llh_means, llh_variances)

        print("Digit: {0}".format(correct_label))
        print("Classified as: {0}".format(label))
        print(comp)
    else:
        testing_digits, testing_labels = ag.io.load_mnist('testing')
        if QUICK:
            testing_digits = testing_digits[:100]
        testing_digits = ag.util.zeropad(testing_digits, (0, 2, 2))
        testing_edges = ag.features.bedges(testing_digits, inflate=inflate, k=bedges_k)

        N = len(testing_edges)
        c = 0
        #all_templates = np.clip(all_templates, eps, 1.0 - eps)
        for i, features in enumerate(testing_edges):
            label, comp = classify(features, all_templates, means, variances, llh_means, llh_variances)
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
