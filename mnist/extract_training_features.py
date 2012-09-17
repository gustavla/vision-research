from __future__ import print_function
import argparse

parser = argparse.ArgumentParser(description='Extract MNIST features (zero-padded to 32 x 32) arranged for training (indexed by label)')
parser.add_argument('dataset', metavar='(training|testing)', type=str, choices=('training', 'testing'), help='Specify training or testing set')
parser.add_argument('output', metavar='<output file>', type=argparse.FileType('wb'), help='Filename of output file')
parser.add_argument('-k', nargs=1, default=[5], choices=range(1, 7), type=int, help='Sensitivity of features. 1-6, with 6 being the most conservative.')
parser.add_argument('-r', dest='range', nargs=2, metavar=('FROM', 'TO'), default=(0, 10000), type=int, help='Range of frames, FROM (incl) and TO (excl)')
#parser.add_argument('--no-inflate', dest='inflate', action='store_false', help='Do not inflate the featured pixels to neigbhors')
parser.add_argument('--save-originals', dest='graylevel', action='store_true', help='Store original graylevel images as well')
parser.add_argument('--radius', metavar='RADIUS', nargs=1, default=[1], type=int, help='Inflation radius')
parser.add_argument('--kernel', metavar='KERNEL', nargs=1, default=['box'], type=str, choices=('box', 'perpendicular'), help='Kernel shape of inflation')

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

if save_graylevel:
    all_digits = np.empty((10, n1-n0, 32, 32), dtype=np.float64)

all_features = np.empty((10, n1-n0, 8, 32, 32), dtype=np.uint8)

min_index = np.inf
max_index = 0


digit_features = {} 
for d in range(10):
    print("Extracting features for digit", d)
    digits, indices = ag.io.load_mnist(dataset, digits=[d], selection=slice(n0, n1), return_labels=False, return_indices=True)
    digits = ag.util.zeropad(digits, (0, 2, 2))

    if save_graylevel:
        all_digits[d] = digits

    min_index = min(min_index, indices[0])
    max_index = max(max_index, indices[-1])

    features = ag.features.bedges(digits, k=k, inflate=inflation_type, radius=inflation_radius)

    #digit_features[str(d)] = features
    all_features[d] = features 

digit_features['features'] = all_features

# Store some meta data
meta = {} 
meta['k'] = k
meta['dataset'] = dataset
meta['range'] = (n0, n1)
meta['index_span'] = (min_index, max_index)
meta['kernel'] = inflation_type 
meta['radius'] = inflation_radius
meta['shape'] = (32, 32)

digit_features['meta'] = meta

if save_graylevel:
    digit_features['originals'] = all_digits 

np.savez(output_file, **digit_features)

