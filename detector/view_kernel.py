
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


fig = plt.figure(figsize=(14,6))
ax = fig.add_subplot(111)

plt.subplot(121)
l2 = plt.imshow(detector.patch_dict.vispatches[0], vmin=0, vmax=1, cmap=plt.cm.gray, interpolation='nearest')

plt.subplot(122)
l = plt.imshow(detector.kernels[mixcomp,...,0], vmin=0, vmax=1, cmap=plt.cm.RdBu, interpolation='nearest')
plt.colorbar()

axindex = plt.axes([0.1, 0.01, 0.8, 0.03], axisbg='lightgoldenrodyellow')
slider = Slider(axindex, 'Index', 0, detector.patch_dict.num_patches-1, valfmt='%1.0f')



def update(val):
    index = slider.val
    #l.set_data(small[...,index])
    l.set_data(detector.kernels[mixcomp,...,index])
    l2.set_data(detector.patch_dict.vispatches[index])
    plt.draw()
slider.on_changed(update)
plt.show()

