from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='Test response of model')
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')
parser.add_argument('img', metavar='<image file>', type=argparse.FileType('rb'), help='Filename of image file')

# TODO: Remove
parser.add_argument('mixcomp', metavar='<mixture component>', type=int, help='mix comp')

args = parser.parse_args()
model_file = args.model
image_file = args.img
mixcomp = args.mixcomp

import gv
import numpy as np
from PIL import Image
import scipy.misc
import matplotlib.pylab as plt

detector = gv.Detector.load(model_file)

img = np.array(Image.open(image_file)).astype(np.float64) / 255.0

def resize(im, factor):
    new_size = tuple([int(round(im.shape[i] * factor)) for i in xrange(2)])
    # TODO: Change to something much more suited for this.
    return scipy.misc.imresize((im*255).astype(np.uint8), new_size).astype(np.float64)/255.0

if 0:
    import time
    start = time.time()
    x, small = detector.response_map(img, mixcomp)
    end = time.time()
    print 'time:', (end - start)

def resize_and_detect(img, mixcomp, factor=1.0):
    img_resized = resize(img, factor)
    x, img_feat = detector.response_map(img_resized, mixcomp)
    return x, img_feat, img_resized

def plot_results(img_resized, x, small, mixcomp, bounding_boxes=[]):
    # Get max peak
    #print ix, iy

    #print '---'
    #print x.shape
    #print small.shape

    plt.clf()
    plt.subplot(221)
    plt.title('Input image')
    plt.imshow(img_resized)
    plt.colorbar()

    for dbb in bounding_boxes:
        bb = dbb.box
        plt.gca().add_patch(plt.Rectangle((bb[1], bb[0]), bb[3]-bb[1], bb[2]-bb[0], facecolor='none', edgecolor='cyan', linewidth=2.0))

    if x is not None:
        plt.subplot(222)
        plt.title('Response map')
        plt.imshow(x, interpolation='nearest')#, vmin=-40000, vmax=-36000)
        plt.colorbar()

    if small is not None:
        plt.subplot(223)
        plt.title('Feature activity')
        plt.imshow(small.sum(axis=-1), interpolation='nearest')
        plt.colorbar()

    plt.subplot(224)
    if 0:
        pass
        plt.title('Normalized stuff')
        plt.imshow(x / np.clip(small.sum(axis=-1), 5, np.inf), interpolation='nearest')
        plt.colorbar()
    else:
        plt.title('Kernel Bernoulli probability averages')
        plt.imshow(detector.kernels[mixcomp].mean(axis=-1), interpolation='nearest', cmap=plt.cm.RdBu, vmin=0, vmax=1)
        plt.colorbar()

if 0:
    #x, small, img_resized = resize_and_detect(img, mixcomp, 0.7)
    print '-'*80
    bbs = detector.detect_coarse_unfiltered_at_scale(img, 0.7, mixcomp)
    print '='*80

    scores = map(lambda x: x[0], bbs)
    plt.hist(scores, 30)
    plt.show()

    bbs.sort(reverse=True)

    print bbs
    print len(bbs)

    #bbs = bbs[-10:]
    print bbs[0]

    #plot_results(img, x, small, mixcomp, bbs)
    plot_results(img, None, None, mixcomp, bbs)
    plt.show()

elif 1:
    bbs = detector.detect_coarse(img, mixcomp) 


    plot_results(img, None, None, mixcomp, bbs)
    plt.show()

else:
    for l, factor in enumerate(np.linspace(0.3, 1.0, 20)):
        x, small, img_resized = resize_and_detect(img, mixcomp, factor)
        plot_results(img_resized, x, small)
        plt.savefig('pic-{0:02}.png'.format(l))

