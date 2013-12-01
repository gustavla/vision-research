
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_model', metavar='<input model file>', type=argparse.FileType('rb'), help='Filename of input model file')

args = parser.parse_args()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
import functools
import gv

imshowi = functools.partial(plt.imshow, interpolation='nearest')

detector = gv.Detector.load(args.input_model)

#EACH = 40


for m in [4]: 
    sh = detector.kernel_templates[m][0].shape
    kp = detector.indices[m][0]
    M = np.zeros(sh)

    for I in kp: 
        M[tuple(I)] = 1

    w = detector.weights(m)

    for f in xrange(detector.num_features):
        plt.clf()
        plt.subplot(211)
        imshowi(M[...,f])
        plt.subplot(212)
        imshowi(w[...,f], vmin=-5, vmax=5, cmap=plt.cm.RdBu_r)
        plt.savefig('output/part{}.png'.format(f))
        
