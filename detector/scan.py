from __future__ import print_function
from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='Test response of model')
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')
parser.add_argument('img_id', metavar='<image id>', type=int, help='ID of image in VOC repository')
parser.add_argument('--single-scale', dest='factor', nargs=1, default=[None], metavar='FACTOR', type=float, help='Run single scale factor')

# TODO: Remove
parser.add_argument('mixcomp', metavar='<mixture component>', nargs='?', type=int, help='mix comp')

args = parser.parse_args()
model_file = args.model
img_id = args.img_id
factor = args.factor[0]
mixcomp = args.mixcomp

import gv
import numpy as np
from PIL import Image
import matplotlib.pylab as plt
import sys
from config import VOCSETTINGS

from plotting import plot_results

detector = gv.Detector.load(model_file)

fileobj = gv.voc.load_training_file(VOCSETTINGS, 'bicycle', img_id)
if fileobj is None:
    print("Could not find image", file=sys.stderr)
    sys.exit(0)
img = gv.img.load_image(fileobj.path)
#img = np.random.random(img.shape)

#print(fileobj)
#sys.exit(0)

#img = np.array(Image.open(image_file)).astype(np.float64) / 255.0

if factor is not None:
    assert mixcomp is not None
    bbs, x, small = detector.detect_coarse_unfiltered_at_scale(img, factor, mixcomp) 
    bbs = detector.nonmaximal_suppression(bbs)

    xx = (x - x.mean()) / x.std()
    plot_results(detector, img, x, small, mixcomp, bbs)
    print('max response', x.max())
    print('max response (xx)', xx.max())
else:
    if mixcomp is None:
        bbs = detector.detect_coarse(img, fileobj=fileobj)
    else:
        bbs = detector.detect_coarse_single_component(img, mixcomp, fileobj=fileobj) 
    print(bbs)
    if len(bbs) > 0:
        print('max score: ', bbs[0].score)
        tot = sum([bb.correct for bb in bbs])
        print("Total: ", tot)
    plot_results(detector, img, None, None, mixcomp, bbs)

print('kernel sum', np.fabs(detector.kernels[mixcomp] - 0.5).sum())

plt.show()

