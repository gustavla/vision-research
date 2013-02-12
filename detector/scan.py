from __future__ import print_function
from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='Test response of model')
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')
parser.add_argument('img_id', metavar='<image id>', type=int, help='ID of image in VOC repository')
parser.add_argument('--kernel-size', dest='side', nargs=1, default=[None], metavar='SIDE', type=float, help='Run single side length of kernel')

# TODO: Make into an option 
parser.add_argument('mixcomp', metavar='<mixture component>', nargs='?', type=int, help='mix comp')

args = parser.parse_args()
model_file = args.model
img_id = args.img_id
side = args.side[0]
mixcomp = args.mixcomp

import gv
import numpy as np
np.seterr(divide='raise')
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

grayscale_img = img.mean(axis=-1)
#img = np.random.random(img.shape)

#print(fileobj)
#sys.exit(0)

#img = np.array(Image.open(image_file)).astype(np.float64) / 255.0

if side is not None:
    assert mixcomp is not None
    #bbs, x, small = detector.detect_coarse_unfiltered_at_scale(grayscale_img, side, mixcomp) 

    factor = side/detector.unpooled_kernel_side
    bbs, x = detector.detect_coarse_single_factor(grayscale_img, factor, mixcomp)
    #bbs = detector.nonmaximal_suppression(bbs)

    #print('small', small.shape)

    print(bbs)
    plot_results(detector, img, x, None, mixcomp, bbs)
    print('max response', x.max())
else:
    if mixcomp is None:
        import time
        start = time.time()
        bbs = detector.detect_coarse(grayscale_img, fileobj=fileobj)
        print("Elapsed:", (time.time() - start))
        #sys.exit(0)
    else:
        bbs = detector.detect_coarse_single_component(grayscale_img, mixcomp, fileobj=fileobj) 
    print(bbs)
    if len(bbs) > 0:
        print('max score: ', bbs[0].score)
        tot = sum([bb.correct for bb in bbs])
        print("Total: ", tot)
    plot_results(detector, img, None, None, mixcomp, bbs)

#print('kernel sum', np.fabs(detector.kernels[mixcomp] - 0.5).sum())

plt.show()

