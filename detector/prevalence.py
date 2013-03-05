from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='Train mixture model on edge data')
parser.add_argument('parts', metavar='<parts file>', type=argparse.FileType('rb'), help='Filename of parts file')


args = parser.parse_args()
parts_file = args.parts

import numpy as np
import matplotlib.pylab as plt
import amitgroup as ag
import glob
import os
import sys
import gv


parts_descriptor = gv.BinaryDescriptor.getclass('parts').load(parts_file)

sett = parts_descriptor.settings

base_path = sett.get('base_path')
path = sett['image_dir']
if base_path is not None:
    path = os.path.join(os.environ[base_path], path)

files = glob.glob(path)

counts = np.zeros(parts_descriptor.num_parts)
tots = 0

def process(f):
    #print "Processing file {0}".format(f)
    im = gv.img.load_image(f)
    edges = parts_descriptor.extract_features(im, {'spread_radii': (0, 0)})

    counts = np.apply_over_axes(np.sum, edges, [0, 1]).ravel() 
    tots = np.prod(edges.shape[1:])
    return counts, tots

from multiprocessing import Pool
p = Pool(7)
res = p.map(process, files)

counts = sum(map(lambda x: x[0], res))
tots = sum(map(lambda x: x[1], res))

plt.hist(np.log(counts / tots)/np.log(10))
plt.xlabel("log10(Features per pixel)")
plt.ylabel("Frequency")
plt.show()
