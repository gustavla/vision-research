from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import argparse
import sys

parser = argparse.ArgumentParser(description='MNIST classifier')
parser.add_argument('features', metavar='<testing features file>', type=argparse.FileType('rb'), help='File with features of testing data (must be arranged as testing features)')
parser.add_argument('mixtures', metavar='<mixtures file>', type=argparse.FileType('rb'), help='Filename of mixtures')
parser.add_argument('coefficients', metavar='<coef file>', type=argparse.FileType('rb'), help='Filename of model coefficients')
parser.add_argument('-p', '--plot', action='store_true', help='Plot in real-time using pygame')
parser.add_argument('-i', dest='inspect', nargs=1, default=[None], metavar='INDEX', type=int, help='Run and inspect a single test index')
parser.add_argument('-r', dest='range', nargs=2, metavar=('FROM', 'TO'), type=int, default=(0, sys.maxint), help='Range of testing indices, FROM (incl) and TO (excl)')
parser.add_argument('-d', '--deform', dest='deform', type=str, choices=('none', 'bernoulli', 'graylevel'), help='What kind of deformations to perform.')
parser.add_argument('--graylevel-deform', action='store_true', help='Use graylevel deformations (image_deformation), and not feature deformations (bernoulli_deformation)')
parser.add_argument('-v', '--verbose', action='store_true', help='Print intermediate information')

args = parser.parse_args()
feat_file = args.features
coef_file = args.coefficients
mixtures_file = args.mixtures
inspect_component = args.inspect[0]
n0, n1 = args.range
PLOT = args.plot
deform_type = args.deform
if deform_type == 'none':
    deform_type = None

verbose = args.verbose

import numpy as np
from classifier import classify, surplus
from scipy.optimize import fmin
import amitgroup as ag
ag.set_verbose(verbose)

features_data = np.load(feat_file)
all_features = features_data['features']
all_labels = features_data['labels'] 
all_templates = np.load(mixtures_file)['templates'] 
coefs = np.load(coef_file)

try:
    all_graylevels = features_data['originals']
    all_graylevel_templates = mixtures_data['graylevel_templates']
except KeyError:
    raise Exception("The feature file must be run with --save-originals")


if (n0, n1) != (0, sys.maxint):
    all_features = all_features[n0:n1]
    all_labels = all_labels[n0:n1]

means = coefs['prior_mean']
variances = coefs['prior_var']
samples = coefs['samples']

if 0:
    #all_templates = np.clip(all_templates, eps, 1.0 - eps)
    
    # Do a search for the best surplus
    def check_neg_surplus(b0):
        print("Running b0 =", b0)
        total_surplus = 0.0
        for i, features in enumerate(all_features):
            label, info = classify(features, all_templates, means, variances, deform=deform_type, correct_label=all_labels[i], b0=b0, lmb0=1e4, debug_plot=PLOT)
            total_surplus += info['surplus_change']
        print("Returning surplus", total_surplus)
        return -total_surplus

    if 0:

        # Now, try to find the minimum value!
        ret = fmin(check_neg_surplus, 0.1, xtol=0.001)

        print(ret)

    if 1:
        N = 50
        bs = np.linspace(0.0, 0.001, N)
        ys = np.empty(N)
        for i, b0 in enumerate(bs):
            total_surplus = -check_neg_surplus(b0) 
            print("b0:", b0, "Total surplus:", total_surplus)
            ys[i] = total_surplus

        np.savez('surplus.npz', b=bs, surplus=ys)

elif inspect_component is not None:
    #testing_digits, testing_labels = ag.io.load_mnist('testing', indices=slice(None, 10))
    #ag.io.load_mnist('testing', indices=inspect_component)
    #digits, labels = ag.io.load_mnist('testing')
    #digits = ag.util.zeropad(digits, (0, 2, 2))
    #digit, correct_label = digits[inspect_component], labels[inspect_component]

    features, correct_label = all_features[inspect_component], all_labels[inspect_component]

    # TODO: Does not work
    label, info = classify(features, all_templates, means, variances, samples, deformation=deform_type, correct_label=correct_label, debug_plot=PLOT)

    print("Digit: {0}".format(correct_label))
    print("Classified as: {0}".format(label))
    print(info['comp'])
else:
    N = len(all_features)
    c = 0
    #all_templates = np.clip(all_templates, eps, 1.0 - eps)
    for i, features in enumerate(all_features):
        additional = {}
        additional['graylevels'] = all_graylevels
        additional['graylevel_templates'] = all_graylevel_templates

        label, info = classify(features, all_templates, means, variances, samples, deformation=use_deformation, correct_label=all_labels[i], debug_plot=PLOT, threshold_multiple=1.3, **additional)
        correct = label == all_labels[i]
        c += correct
        print(i, N, correct)

        if False and not correct:
            #print("Label = {0}, component = {1}".format(label, comp))
            ag.plot.images(testing_digits[i])
            plt.show()
            ag.plot.images(np.rollaxis(features, 2))
            plt.show()
            ag.plot.images(np.rollaxis(all_templates[label,info['comp']], 2))
            plt.show()

    print("Success rate: {0:.2f} ({1}/{2})".format(100*c/N, c, N))
