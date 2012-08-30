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
parser.add_argument('-d', '--deform', dest='deform', type=str, choices=('none', 'edges', 'intensity'), help='What kind of features to perform the deformations on.')
#parser.add_argument('--graylevel-deform', action='store_true', help='Use graylevel deformations (image_deformation), and not feature deformations (bernoulli_deformation)')
parser.add_argument('-v', '--verbose', action='store_true', help='Print intermediate information')
parser.add_argument('-a', dest='alpha', metavar='ALPHA', nargs=1, default=[1.3], type=float, help='Selective deformation threshold multiple')
parser.add_argument('--test-conjugate', nargs=4, metavar=('BFROM', 'BTO', 'ETA', 'N'), default=(None, None, None, 0), type=float, help='Test various values of beta and eta and store in surplus.npz')

args = parser.parse_args()
feat_file = args.features
coef_file = args.coefficients
mixtures_file = args.mixtures
inspect_component = args.inspect[0]
n0, n1 = args.range
PLOT = args.plot
deform_type = args.deform
alpha = args.alpha[0]
bfrom, bto, eta, bN = args.test_conjugate
bN = int(bN)
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
mixtures_data = np.load(mixtures_file)
all_templates = mixtures_data['templates'] 
coefs = np.load(coef_file)

try:
    all_graylevels = features_data['originals']
    all_graylevel_templates = mixtures_data['graylevel_templates']
except KeyError:
        raise Exception("The feature file must be run with --save-originals, and the mixtures must be trained with this file")


if (n0, n1) != (0, sys.maxint):
    all_features = all_features[n0:n1]
    all_labels = all_labels[n0:n1]

means = coefs['prior_mean']
variances = coefs['prior_var']
samples = coefs['samples']

if bN:
    #all_templates = np.clip(all_templates, eps, 1.0 - eps)
    
    # Do a search for the best surplus
    def check_neg_surplus(b0):
        print("Running b0 =", b0)
        total_surplus = 0.0
        for i, features in enumerate(all_features):
            label, info = classify(features, all_templates, means, variances, deformation=deform_type, correct_label=all_labels[i], b0=b0, lmb0=100, samples=samples, debug_plot=PLOT, threshold_multiple=alpha)
            total_surplus += info['surplus_change']
        print("Returning surplus", total_surplus)
        return -total_surplus

    if 0:

        # Now, try to find the minimum value!
        ret = fmin(check_neg_surplus, 0.1, xtol=0.001)

        print(ret)

    if 1:
        bs = np.linspace(bfrom, bto, bN)
        ys = np.empty(bN)
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
    num_deformed = 0
    num_contendors = 0
    incorrect_and_undeformed = 0
    turned_correct = 0
    turned_incorrect = 0
    #all_templates = np.clip(all_templates, eps, 1.0 - eps)
    for i, features in enumerate(all_features):
        additional = {}
        additional['graylevels'] = all_graylevels[i]
        additional['graylevel_templates'] = all_graylevel_templates

        label, info = classify(features, all_templates, means, variances, samples=samples, deformation=deform_type, correct_label=all_labels[i], debug_plot=PLOT, threshold_multiple=alpha, **additional)
        correct = label == all_labels[i]
        c += correct
        num_deformed += info['deformation']
        if info['deformation']:
            num_contendors += info['num_contendors']
            turned_correct += info['turned_correct']
            turned_incorrect += info['turned_incorrect']
        

        if not correct and not info['deformation']:
            # It was not correct and no deformation was made. Alpha might be too low!
            incorrect_and_undeformed += 1
        print(i, N, correct)

        if False and not correct:
            #print("Label = {0}, component = {1}".format(label, comp))
            ag.plot.images(testing_digits[i])
            plt.show()
            ag.plot.images(np.rollaxis(features, 2))
            plt.show()
            ag.plot.images(np.rollaxis(all_templates[label,info['comp']], 2))
            plt.show()


    print("Deformed: {0:.2f}%".format(100*num_deformed/N))
    if num_deformed > 0:
        print("Average contendors: {0:.2f}".format(num_contendors/num_deformed))
    print("Incorrect and undeformed: {0:.2f}%".format(100*incorrect_and_undeformed/N))
    print("Turned correct, incorrect: {0:.2f}%, {1:.2f}%".format(100*turned_correct/N, 100*turned_incorrect/N))
    print("Miss rate: {0:.2f}% ({1}/{2})".format(100*(N-c)/N, (N-c), N))
