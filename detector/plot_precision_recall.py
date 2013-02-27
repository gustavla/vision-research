from __future__ import division
from __future__ import print_function
import argparse

parser = argparse.ArgumentParser(description='Test response of model')
parser.add_argument('results', metavar='<file>', nargs='+', type=argparse.FileType('rb'), help='Filename of results file')
parser.add_argument('--captions', metavar='<caption>', nargs='*', type=str, help='Captions')

args = parser.parse_args()
results_files = args.results
captions = args.captions
if captions:
    assert len(captions) == len(results_files), "Must supply caption for all"

import numpy as np
import matplotlib.pylab as plt

for i, results_file in enumerate(results_files):
    data = np.load(results_file)

    p, r = data['precisions'], data['recalls']
    detections = data['detections']
    ap = data['ap']

    p = np.r_[[1], p]
    r = np.r_[[0], r]

    print(results_file.name)
    print('AP:', ap)
    print(detections[-10:])
    print()

    if captions:
        caption = captions[i]
    else:
        caption = results_file.name
    plt.plot(r, p, label=caption)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim((0, 1))
    plt.ylim((0, 1))

plt.legend()
plt.show()

