
from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='View mixture components')
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')
parser.add_argument('imgname', metavar='<image name>', type=int, help='Name of image in VOC repository')
parser.add_argument('mixcomp', metavar='<mixture component>', type=int, help='mix comp')
parser.add_argument('scale', metavar='<scale>', type=float, help='Scale factor')
parser.add_argument('x', metavar='<x>', type=int, help='X offset')
parser.add_argument('y', metavar='<y>', type=int, help='Y offset')

args = parser.parse_args()
model_file = args.model
imagename = args.imgname
mixcomp = args.mixcomp
factor = args.scale
offset = args.x, args.y

import gv
import numpy as np
import matplotlib.pylab as plt
from matplotlib.widgets import Slider
from config import VOCSETTINGS
from plotting import plot_box
import amitgroup as ag

# Load detector
detector = gv.Detector.load(model_file)

# Load image
fileobj = gv.voc.load_training_file(VOCSETTINGS, 'bicycle', imagename)
img = gv.img.load_image(fileobj.path)

# Size of the kernel
sh = detector.kernels.shape[1:3]

# Run detection (mostly to get resized image)
x, small, img_resized = detector.resize_and_detect(img, mixcomp, factor)

pooling_size = detector.patch_dict.settings['pooling_size']
box = [offset[0] * pooling_size[0], 
       offset[1] * pooling_size[1], 
       (offset[0]+sh[0]) * pooling_size[0], 
       (offset[1]+sh[1]) * pooling_size[1]]

box = map(lambda x: x / factor, box)

img_padded = ag.util.zeropad(img_resized, (sh[0], sh[1], 0))

print img.shape
print img_resized.shape
print img_padded.shape

edges = ag.features.bedges_from_image(img_resized, **detector.patch_dict.bedges_settings())
small = detector.patch_dict.extract_pooled_parts(edges)
print edges.shape
print small.shape
#img_kernel = img_padded[sh[0] + offset[0] : sh[0] + offset[0] + sh[0], sh[1] + offset[1] : sh[1] + offset[1] + sh[1]]

small_padded = ag.util.zeropad(small, (sh[0], sh[1], 0))

window = small_padded[sh[0] + offset[0] : sh[0] + offset[0] + sh[0], sh[1] + offset[1] : sh[1] + offset[1] + sh[1]]

print window.shape

# Get background
back = np.zeros(small.shape[-1])
for f in xrange(small.shape[-1]):
    back[f] = small[...,f].sum() / np.prod(small.shape[:2])

back = np.clip(back, 0.01, 0.99)

plt.subplot(231)
plt.imshow(img)
plot_box(box)

plt.subplot(232)
l1 = plt.imshow(np.ones((1, 1)), vmin=0, vmax=1, cmap=plt.cm.gray, interpolation='nearest')
plt.title("Window")

plt.subplot(233)
l4 = plt.imshow(np.ones((1, 1)), vmin=0, vmax=1, cmap=plt.cm.gray, interpolation='nearest')
plt.title("Edge visualization")

plt.subplot(234)
l2 = plt.imshow(np.ones((1, 1)), vmin=0, vmax=1, cmap=plt.cm.RdBu, interpolation='nearest')
plt.colorbar()
plt.title("Kernel")

plt.subplot(235)
l3 = plt.imshow(np.ones((1,1)), vmin=-4, vmax=4, cmap=plt.cm.RdBu, interpolation='nearest')
plt.title("Contribution")
plt.colorbar()

kernels = detector.kernels.copy()
#for f in xrange(small.shape[-1]):
    #kernels[...,f] = np.clip(kernels[...,f], back[0,0,f], 1.0-self.back[0,0,f])

bk = (detector.small_support < 0.1).astype(float)
ss = detector.small_support.copy()
ss *= 5 
ss = np.clip(ss, 0, 1)
for f in xrange(small.shape[-1]):
    #kernels[...,f] = np.clip(kernels[...,f], back[f], 1.0-back[f])
    #kernels[...,f] = bk * np.clip(kernels[...,f], back[f], 1.0-back[f]) + (1.0-bk) * kernels[...,f]
    kernels[...,f] /= np.clip(detector.small_support, 0.3, 1.0)
    #kernels[...,f] = (1-ss) * np.clip(kernels[...,f], back[f], 1.0-back[f]) + ss * kernels[...,f]
    kernels[...,f] = np.clip((1-ss) * back[f] + ss * kernels[...,f], 0.05, 0.95)


# What is the score if this kernel is convolved with complete background?
score = (np.log(1.0 - kernels[mixcomp]) - np.log(1.0 - back)).sum()
#for index in xrange(small.shape[-1]):
    #data = \
        #np.log(1.0 - kernels[mixcomp,...,index]) - \
        ##np.log(1.0 - back[index])
    #score += data.sum()
print "Back avarage:", back.mean()
print "Back score:", score

contribution_map = np.zeros(sh) 
for index in xrange(small.shape[-1]):
    data = \
        window[...,index] * np.log(kernels[mixcomp,...,index]) + \
        (1.0-window[...,index]) * np.log(1.0 - kernels[mixcomp,...,index]) + \
        (-1) * (1.0-window[...,index]) * np.log(1.0 - back[index]) + \
        (-1) * window[...,index] * np.log(back[index])

    contribution_map += data

ext = max(-contribution_map.min(), contribution_map.max())
plt.subplot(236)
plt.imshow(contribution_map, vmin=-ext, vmax=ext, cmap=plt.cm.RdBu, interpolation='nearest')
plt.colorbar()
print "Total contribution", contribution_map.sum()

axindex = plt.axes([0.1, 0.01, 0.8, 0.03], axisbg='lightgoldenrodyellow')
slider = Slider(axindex, 'Index', 0, detector.patch_dict.num_patches-1, valfmt='%1.0f')

#r3 = masked_convolve(1-bigger, -self.log_invback)
#r4 = masked_convolve(bigger, -self.log_back)

def update(val):
    index = int(val)
    l1.set_data(window[...,index])
    l2.set_data(kernels[mixcomp,...,index])
    data = \
        window[...,index] * np.log(kernels[mixcomp,...,index]) + \
        (1.0-window[...,index]) * np.log(1.0 - kernels[mixcomp,...,index]) + \
        (-1) * (1.0-window[...,index]) * np.log(1.0 - back[index]) + \
        (-1) * window[...,index] * np.log(back[index])
    l3.set_data(data)
    l4.set_data(detector.patch_dict.vispatches[index])

    print '({0}) Contribution: {1} (back: {2})'.format(index, data.sum(), back[index])
    plt.draw()
slider.on_changed(update)

update(0)
plt.show()
