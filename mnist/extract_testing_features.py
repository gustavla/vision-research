from __future__ import print_function
import argparse

parser = argparse.ArgumentParser(description='Extract MNIST features (zero-padded to 32 x 32) arranged for testing (including label)')
parser.add_argument('dataset', metavar='(training|testing)', type=str, choices=('training', 'testing'), help='Specify training or testing set')
parser.add_argument('output', metavar='<output file>', type=argparse.FileType('wb'), help='Filename of output file')
parser.add_argument('-k', nargs=1, default=[5], choices=range(1, 7), type=int, help='Sensitivity of features. 1-6, with 6 being the most conservative.')
parser.add_argument('-r', dest='range', nargs=2, metavar=('FROM', 'TO'), default=(0, -1), type=int, help='Range of frames, FROM (incl) and TO (excl)')
parser.add_argument('--save-originals', dest='graylevel', action='store_true', help='Store original graylevel images as well')
parser.add_argument('--radius', metavar='RADIUS', nargs=1, default=[1], type=int, help='Inflation radius')
parser.add_argument('--kernel', metavar='KERNEL', nargs=1, default=['box'], type=str, choices=('box', 'orthogonal'), help='Kernel shape of inflation')

args = parser.parse_args()
dataset = args.dataset
output_file = args.output
k = args.k[0]
n0, n1 = args.range
inflation_radius = args.radius[0]
inflation_type = args.kernel[0]
save_graylevel = args.graylevel
    
#assert dataset in ('training', 'testing')

import numpy as np
import amitgroup as ag
import sys
import os

digit_features = {}

digits, labels = ag.io.load_mnist(dataset, selection=slice(n0, n1))

#digits_padded = np.zeros((len(digits),) + (32, 32))
#digits_padded[:,2:-2,2:-2] = digits
digits_padded = ag.util.zeropad(digits, (0, 2, 2))

features = ag.features.bedges(digits_padded, k=k, spread=inflation_type, radius=inflation_radius, first_axis=True)
digit_features["features"] = features
digit_features["labels"] = labels

# Store some meta data
meta = {}
meta['k'] = k
meta['dataset'] = dataset
meta['range'] = (n0, n1)
meta['kernel'] = inflation_type 
meta['radius'] = inflation_radius
meta['shape'] = (32, 32)

digit_features['meta'] = meta

if save_graylevel:
    digit_features['originals'] = digits_padded

np.savez(output_file, **digit_features)

