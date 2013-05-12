from __future__ import print_function
from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='Test response of model')
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')
parser.add_argument('img_id', metavar='<image id>', type=int, help='ID of image in VOC repository')
parser.add_argument('--class', dest='obj_class', nargs=1, default=[None], type=str, help='Object class for marking corrects')
parser.add_argument('--kernel-size', dest='side', nargs=1, default=[None], metavar='SIDE', type=float, help='Run single side length of kernel')
parser.add_argument('--contest', type=str, choices=('voc', 'uiuc', 'uiuc-multiscale'), default='voc', help='Contest to try on')
parser.add_argument('--limit', nargs=1, default=[None], type=int, help='Contest to try on')

# TODO: Make into an option 
parser.add_argument('mixcomp', metavar='<mixture component>', nargs='?', type=int, help='mix comp')

args = parser.parse_args()
model_file = args.model
img_id = args.img_id
side = args.side[0]
mixcomp = args.mixcomp
obj_class = args.obj_class[0]
contest = args.contest
bb_limit = args.limit[0]

import gv
import numpy as np
import matplotlib.pylab as plt
import sys
from config import VOCSETTINGS
import skimage.data

from plotting import plot_results

detector = gv.Detector.load(model_file)

print(obj_class)

if contest == 'voc':
    fileobj = gv.voc.load_training_file(VOCSETTINGS, obj_class, img_id)
elif contest == 'uiuc':
    assert obj_class is None or obj_class == 'car', "Can't set object class for uiuc data"
    fileobj = gv.uiuc.load_testing_file(img_id)

elif contest == 'uiuc-multiscale':
    assert obj_class is None or obj_class == 'car', "Can't set object class for uiuc data"
    fileobj = gv.uiuc.load_testing_file(img_id, single_scale=False)

if fileobj is None:
    print("Could not find image", file=sys.stderr)
    sys.exit(0)
img = skimage.data.load(fileobj.path).astype(np.float64)/255
grayscale_img = gv.img.asgray(img)
print("Image size:", grayscale_img.shape)
#img = np.random.random(img.shape)

#print(fileobj)
#sys.exit(0)

if side is not None:
    assert mixcomp is not None
    #bbs, x, small = detector.detect_coarse_unfiltered_at_scale(grayscale_img, side, mixcomp) 

    factor = side/max(detector.orig_kernel_size)
    print(factor)
    bbs, x, feats, img_resized = detector.detect_coarse_single_factor(grayscale_img, factor, mixcomp)
    #bbs = detector.nonmaximal_suppression(bbs)

    #print('small', small.shape)

    for bb in bbs:
        print(bb)
    plot_results(detector, img, x, feats, mixcomp, bbs, img_resized=img_resized)
    print('max response', x.max())
else:
    if mixcomp is None:
        import time
        start = time.time()
        bbs = detector.detect_coarse(grayscale_img, fileobj=fileobj)
        print("Elapsed:", (time.time() - start))
        #sys.exit(0)
    else:
        bbs = detector.detect_coarse(grayscale_img, fileobj=fileobj, mixcomps=[mixcomp]) 

    if bb_limit is not None:
        bbs = bbs[:bb_limit]

    for bb in bbs:
        print(bb)
    if len(bbs) > 0:
        print('max score: ', bbs[0].score)
        tot = sum([bb.correct for bb in bbs])
        print("Total: ", tot)
    plot_results(detector, img, None, None, mixcomp, bbs)

#print('kernel sum', np.fabs(detector.kernels[mixcomp] - 0.5).sum())

plt.show()

