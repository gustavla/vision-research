
import argparse

parser = argparse.ArgumentParser(description='Train mixture model on edge data')
parser.add_argument('parts', metavar='<parts file>', type=argparse.FileType('rb'), help='Filename of parts file')
parser.add_argument('--all', action='store_true', default=False)
parser.add_argument('-o', '--output', type=argparse.FileType('wb'), help='Filename of output image')

args = parser.parse_args()
parts_file = args.parts


import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
import amitgroup as ag
import amitgroup.features
import sys
import os
import gv

#parts_dictionary = gv.PatchDictionary.load(part_file)
descriptor = gv.BinaryDescriptor.getclass('parts').load(parts_file)

originals = descriptor.visparts
parts = descriptor.parts
E = parts.shape[-1]

diff = parts.shape[0] / descriptor.num_true_parts
if args.all:
    F = parts.shape[0] 
else:
    F = descriptor.num_true_parts

#F = 50

plt.figure(figsize=(10, 10))
shape = (F, E + 1)

grid = gv.plot.ImageGrid(shape[0], shape[1], parts.shape[1:3], border_color=(0.3, 0, 0))

strides = descriptor.parts.shape[0] // F

for f in xrange(F):
    #plt.subplot(*(shape + (1 + shape[1] * f,)))
    #plt.imshow(descriptor.visparts[f], interpolation='nearest', cmap=plt.cm.gray)
    if args.all:
        if f % diff == 0:
            grid.set_image(descriptor.visparts[f // diff], f, 0, cmap=plt.cm.gray) 
    else:
        grid.set_image(descriptor.visparts[f], f, 0, cmap=plt.cm.gray) 

    for e in xrange(E):
        #plt.subplot(*(shape + (1 + shape[1] * f + e,)))
        #plt.imshow(descriptor.parts[f,...,e], vmin=0, vmax=1, interpolation='nearest', cmap=plt.cm.RdBu_r)
        grid.set_image(descriptor.parts[strides*f,...,e], f, 1+e, vmin=0, vmax=1, cmap=plt.cm.RdBu_r)

grid.save(args.output, scale=3)
os.chmod(args.output.name, 0644)
#ag.plot.images(originals, zero_to_one=False)
#plt.show()

#ag.plot.images(np.rollaxis(parts[9], axis=2))
