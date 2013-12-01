
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_model', metavar='<input model file>', type=argparse.FileType('rb'), help='Filename of input model file')
parser.add_argument('output_model', metavar='<output model file>', type=argparse.FileType('wb'), help='Filename of model model file')

args = parser.parse_args()

import numpy as np
import gv
from scipy import signal

detector = gv.Detector.load(args.input_model)

def _gauss_kern(sig, size):
    """ Returns a normalized 2D gauss kernel array for convolutions. """
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = np.exp(-(x**2/(2*sig**2)+y**2/(2*sig**2)))
    return g / g.sum()

sharpen_kernel = 1.75*_gauss_kern(0.001, 15) - 0.75*_gauss_kern(5, 15)

eps = detector.settings['min_probability']

for i in xrange(detector.num_mixtures):
    for f in xrange(detector.descriptor.num_features):
        im = detector.kernel_templates[i][0][...,f]# - detector.fixed_spread_bkg[i][0][...,f]

        new_kern = signal.convolve2d(im, sharpen_kernel, mode='same')

        #detector.fixed_spread_bkg[i][0][...,f] + 
        detector.kernel_templates[i][0][...,f] = np.clip(new_kern, eps, 1 - eps)

detector.save(args.output_model)
