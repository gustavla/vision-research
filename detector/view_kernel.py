
from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='View mixture components')
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')
parser.add_argument('mixcomp', metavar='<mixture component>', type=int, help='mix comp')

args = parser.parse_args()
model_file = args.model
mixcomp = args.mixcomp

import gv
import numpy as np
import matplotlib.pylab as plt
from matplotlib.widgets import Slider

detector = gv.Detector.load(model_file)

image = gv.img.load_image('building.png')
back, kernels, small = detector.prepare_kernels(image, mixcomp)

fig = plt.figure(figsize=(14,6))
ax = fig.add_subplot(111)

# TODO: Broken after refactoring

# Check if PartsDescriptor, which is required here.

plt.subplot(131)
l2 = plt.imshow(detector.descriptor.visparts[0], vmin=0, vmax=1, cmap=plt.cm.gray, interpolation='nearest')

plt.subplot(132)
l = plt.imshow(kernels[mixcomp,...,0], vmin=0, vmax=1, cmap=plt.cm.RdBu, interpolation='nearest')
plt.colorbar()

plt.subplot(133)
l3 = plt.imshow(detector.support[mixcomp], vmin=0, vmax=1, cmap=plt.cm.gray, interpolation='nearest')
plt.colorbar()

axindex = plt.axes([0.1, 0.01, 0.8, 0.03], axisbg='lightgoldenrodyellow')
slider = Slider(axindex, 'Index', 0, detector.descriptor.num_parts-1, valfmt='%1.0f')



def update(val):
    index = slider.val
    #l.set_data(small[...,index])
    l.set_data(kernels[mixcomp,...,index])
    l2.set_data(detector.descriptor.visparts[index])
    plt.draw()
slider.on_changed(update)
plt.show()

