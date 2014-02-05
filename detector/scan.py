from __future__ import print_function
from __future__ import division
import argparse
import gv

parser = argparse.ArgumentParser(description='Test response of model')
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')
parser.add_argument('img_id', metavar='<image id>', type=str, help='ID of image in VOC repository')
parser.add_argument('--class', dest='obj_class', nargs=1, default=[None], type=str, help='Object class for marking corrects')
parser.add_argument('--kernel-size', dest='side', nargs=1, default=[None], metavar='SIDE', type=float, help='Run single side length of kernel')
parser.add_argument('--contest', type=str, choices=gv.datasets.datasets(), default='voc-val', help='Contest to try on')
parser.add_argument('--image-file', type=str, nargs=1, default=[None])
parser.add_argument('--limit', nargs=1, default=[None], type=int, help='Limit bounding boxes')
parser.add_argument('--param', type=float, default=None)
parser.add_argument('--filter', type=str, default=None, help='Add filter to make detection harder')

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
img_file = args.image_file[0]

import gv
import numpy as np
import matplotlib.pylab as plt
import sys

from plotting import plot_results

detector = gv.Detector.load(model_file)
#detector.TEMP_second = True
detector._param = args.param

fileobj = gv.datasets.load_file(contest, img_id, obj_class=obj_class, path=img_file)

#elif contest == 'voc':
#    fileobj = gv.voc.load_file(obj_class, img_id)
#elif contest == 'uiuc':
#    assert obj_class is None or obj_class == 'car', "Can't set object class for uiuc data"
#    fileobj = gv.uiuc.load_testing_file(img_id)
#
#elif contest == 'uiuc-multiscale':
#    assert obj_class is None or obj_class == 'car', "Can't set object class for uiuc data"
#    fileobj = gv.uiuc.load_testing_file(img_id, single_scale=False)

if fileobj is None:
    print("Could not find image", file=sys.stderr)
    sys.exit(0)
img = gv.img.load_image(fileobj.path) 
grayscale_img = gv.img.asgray(img)
grayscale_img = gv.imfilter.apply_filter(grayscale_img, args.filter)
print("Image size:", grayscale_img.shape)
#img = np.random.random(img.shape)

if args.filter is not None:
    img = grayscale_img

#print(fileobj)
#sys.exit(0)

if side is not None:
    assert mixcomp is not None
    #bbs, x, small = detector.detect_coarse_unfiltered_at_scale(grayscale_img, side, mixcomp) 

    factor = side/max(detector.settings['image_size'])
    print(factor)
    bbs, x, bkgcomp, feats, img_resized = detector.detect_coarse_single_factor(grayscale_img, factor, mixcomp)
    detector.label_corrects(bbs, fileobj)
    #bbs = detector.nonmaximal_suppression(bbs)

    #print('small', small.shape)
    if bb_limit is not None:
        bbs = bbs[:bb_limit]

    for bb in bbs:
        print(bb)
    plot_results(detector, img, x, feats, mixcomp, bbs, img_resized=img_resized)
    import pylab as plt
    plt.show()
    #plt.imshow(bkgcomp, interpolation='nearest'); plt.colorbar(); plt.show()
    print('max response', x.max())
else:
    if mixcomp is None:
        import time
        start = time.time()
        bbs = detector.detect_coarse(grayscale_img, fileobj=fileobj)
        print('score', bbs[0].score)
        print("Elapsed:", (time.time() - start))
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

