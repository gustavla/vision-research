from __future__ import print_function
import argparse

parser = argparse.ArgumentParser(description='Extract MNIST features (zero-padded to 32 x 32) arranged for testing (including label)')
parser.add_argument('dataset', metavar='(training|testing)', type=str, choices=('training', 'testing'), help='Specify training or testing set')
parser.add_argument('output', metavar='<output file>', type=argparse.FileType('wb'), help='Filename of output file')
parser.add_argument('-k', nargs=1, default=[5], choices=range(1, 7), type=int, help='Sensitivity of features. 1-6, with 6 being the most conservative.')
parser.add_argument('-r', dest='range', nargs=2, metavar=('FROM', 'TO'), default=(0, -1), type=int, help='Range of frames, FROM (incl) and TO (excl)')
parser.add_argument('--no-inflate', dest='inflate', action='store_false', help='Do not inflate the featured pixels to neigbhors')

args = parser.parse_args()
dataset = args.dataset
output_file = args.output
k = args.k[0]
n0, n1 = args.range
inflate = args.inflate
    
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

features = ag.features.bedges(digits_padded, k=k, inflate=inflate)
digit_features["features"] = features
digit_features["labels"] = labels

# Store some meta data
meta = {}
meta['k'] = k
meta['dataset'] = dataset
meta['range'] = (n0, n1)
meta['inflate'] = inflate
meta['shape'] = (32, 32)

digit_features['meta'] = meta

np.savez(output_file, **digit_features)

