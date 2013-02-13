from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='Test response of model')
parser.add_argument('results', metavar='<file>', type=argparse.FileType('rb'), help='Filename of results file')

args = parser.parse_args()
results_file = args.results

import numpy as np
import matplotlib.pylab as plt

data = np.load(results_file)

p, r = data['precisions'], data['recalls']
ap = data['ap']

p = np.r_[[1], p]
r = np.r_[[0], r]

print p
print r

print 'AP:', ap

plt.plot(r, p)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.show()
