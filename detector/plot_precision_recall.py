from __future__ import division
from __future__ import print_function
import argparse

parser = argparse.ArgumentParser(description='Test response of model')
parser.add_argument('results', metavar='<file>', nargs='+', type=argparse.FileType('rb'), help='Filename of results file')

args = parser.parse_args()
results_files = args.results

import numpy as np
import matplotlib.pylab as plt

for results_file in results_files:
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

    plt.plot(r, p, label=results_file.name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim((0, 1))
    plt.ylim((0, 1))

plt.legend()
plt.show()

