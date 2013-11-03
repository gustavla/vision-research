
from __future__ import division

import numpy as np
import scipy.special 

rs = np.random.RandomState()

EM = 0.57721

amount = np.asarray([1000000, 100000])
densities = np.asarray([1/10, 1/10])
means = np.asarray([0.0, 0.8])

all_samples = [] 

for i, N in enumerate(amount):
    samples = np.vstack([rs.normal(means[i], size=N), i*np.ones(N)])
    
    all_samples.append(samples)

all_s = np.hstack(all_samples)

II = np.argsort(all_s)

all_s = all_s[:,II[0]]

top_samples = all_s[:,-10000:]

counts = np.bincount(top_samples[1].astype(np.int32), minlength=len(amount))

print counts * densities 

import pylab as plt

bins = np.linspace(-10, 10, 200)

if 0:
    plt.hist(all_samples[0][0], bins=bins, alpha=0.5)
    plt.hist(all_samples[1][0], bins=bins, alpha=0.5)
    plt.ylim((0, 1000))
    plt.show()
#samples = 
