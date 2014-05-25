from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='Test response of model')
parser.add_argument('results', metavar='<file>', type=argparse.FileType('rb'), help='Filename of results file')
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')

args = parser.parse_args()
results_file = args.results
model_file = args.model

import numpy as np
import gv

detector = gv.Detector.load(model_file)

data = np.load(results_file)

#p, r = data['precisions'], data['recalls']
detections = data['detections']

detections.sort(order='img_id')

N = len(detections)

all_llhs = [[] for i in xrange(20)]
max_mixcomp = -1 

for i, d in enumerate(detections):
    mixcomp = d['mixcomp']
    if mixcomp > max_mixcomp:
        max_mixcomp = mixcomp

    conf = d['confidence']# * detector.fixed_train_std[mixcomp] + detector.fixed_train_mean[mixcomp]

    all_llhs[mixcomp].append(conf)

from pylab import *

all_llhs = all_llhs[:max_mixcomp+1]

ra = (-1000, 8000)
dt = 250
#ra = (-3, 20)
#dt = 0.25
ra = (-1.5, 1.5)
dt = 0.05

print 'detections:', map(np.shape, all_llhs)

for i, llhs in enumerate(all_llhs):
    subplot(len(all_llhs)//2, 2, 1+i)
    hist(llhs, normed=True, bins=np.arange(ra[0], ra[1], dt))
    xlim(ra)
    mu, sigma = np.mean(llhs), np.std(llhs)
    print mu, sigma
    x = np.linspace(ra[0], ra[1], 100)
    y = np.exp(-(x - mu)**2 / (2*sigma**2)) / np.sqrt(2*np.pi*sigma**2)
    plot(x, y, color='red')

show()
