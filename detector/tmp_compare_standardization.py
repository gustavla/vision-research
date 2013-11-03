from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='Compare standardization')
parser.add_argument('models', metavar='<model file>', nargs='+', type=argparse.FileType('rb'), help='Filename of model file')

args = parser.parse_args()

import gv
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.rc('font', size=8) 

import matplotlib.pylab as plt

detectors = [gv.Detector.load(mfile) for mfile in args.models]

M = detectors[0].num_mixtures

for m in xrange(M):
    plt.subplot(M, 1, 1+m)

    for i, det in enumerate(detectors):
        mu = det.standardization_info[m][0]['mean']
        std = det.standardization_info[m][0]['std']

        x = np.linspace(mu - std*5, mu + std*5, 100)
        plt.plot(x, norm.pdf(x, loc=mu, scale=std), label='{}'.format(args.models[i].name))

plt.legend(fontsize='xx-small', framealpha=0.2)
plt.show()
