from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='Compares bkg and obj model')
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')

args = parser.parse_args()
model_file = args.model

import matplotlib.pylab as plt
import numpy as np
import gv

detector = gv.Detector.load(model_file)

bkg_spread = detector.bkg_model(None, spread=True)

psize = detector.settings['subsample_size']
radii = detector.settings['spread_radii']

sub_kernels = detector.prepare_kernels(None, settings=dict(spread_radii=radii, subsample_size=psize))

bins = np.arange(-3, 0, 0.1)

mixcomp = 0

flattened = sub_kernels[mixcomp].ravel()
import pdb; pdb.set_trace()

plt.subplot(211)
plt.hist(np.log(bkg_spread)/np.log(10), bins, normed=True)
plt.title('Background model')
plt.xlim((-3, 0))
plt.ylim((0, 5))
plt.subplot(212)
plt.hist(np.log(flattened)/np.log(10), bins, normed=True)
plt.title('Object model')
plt.xlim((-3, 0))
plt.ylim((0, 5))
plt.xlabel('Probability of feature (log10)')
plt.show()

plt.plot(bkg_spread, label='bkg')
plt.plot(sub_kernels[mixcomp].reshape((-1, detector.num_features)).mean(axis=0), label='obj')
plt.legend()
plt.show()
