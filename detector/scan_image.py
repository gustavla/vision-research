from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='Test response of model')
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')
parser.add_argument('img', metavar='<image file>', type=argparse.FileType('rb'), help='Filename of image file')
parser.add_argument('--single-scale', dest='factor', nargs=1, default=[None], metavar='FACTOR', type=float, help='Run single scale factor')

# TODO: Remove
parser.add_argument('mixcomp', metavar='<mixture component>', type=int, help='mix comp')

args = parser.parse_args()
model_file = args.model
image_file = args.img
factor = args.factor[0]
mixcomp = args.mixcomp

import gv
import numpy as np
from PIL import Image
import matplotlib.pylab as plt

from plotting import plot_results

detector = gv.Detector.load(model_file)

img = np.array(Image.open(image_file)).astype(np.float64) / 255.0

if factor is not None:
    bbs, x, small = detector.detect_coarse_unfiltered_at_scale(img, factor, mixcomp) 
    xx = (x - x.mean()) / x.std()
    plot_results(detector, img, xx, small, mixcomp, bbs)
    print 'max response', x.max()
    print 'max response (xx)', xx.max()
else:
    bbs = detector.detect_coarse(img, mixcomp) 
    plot_results(detector, img, None, None, mixcomp, bbs)

print 'kernel sum', np.fabs(detector.kernels[mixcomp] - 0.5).sum()

plt.show()

