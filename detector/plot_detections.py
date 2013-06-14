from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='Test response of model')
parser.add_argument('results', metavar='<file>', type=argparse.FileType('rb'), help='Filename of results file')

args = parser.parse_args()
results_file = args.results

import numpy as np
import matplotlib.pylab as plt

data = np.load(results_file)

#p, r = data['precisions'], data['recalls']
detections = data['detections']

detections.sort(order='img_id')

N = len(detections)

x = np.zeros(N)
y = np.zeros(N)
c = ['g' for i in xrange(N)] 

for i, d in enumerate(detections):
    x[i] = d['img_id'] 
    x[i] += 20000 * d['mixcomp']
    y[i] = d['confidence']
    c[i] = ['r', 'g'][d['correct']]

plt.scatter(x, y, c=c, s=25)
plt.show()
