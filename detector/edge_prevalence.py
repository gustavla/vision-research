from __future__ import division
from settings import argparse_settings
sett = argparse_settings("Check prevalence")
esettings = sett['edges']
dsettings = sett['parts']

import numpy as np
import matplotlib.pylab as plt
import amitgroup as ag
import glob
import os
import sys
import gv


descriptor = gv.BinaryDescriptor.getclass('edges').load_from_dict(esettings)

base_path = dsettings.get('base_path')
path = dsettings['image_dir']
if base_path is not None:
    path = os.path.join(os.environ[base_path], path)

files = glob.glob(path)[:1]

counts = np.zeros(8)
tots = 0

def process(f):
    #print "Processing file {0}".format(f)
    im = gv.img.load_image(f)
    edges = descriptor.extract_features(im, {'radius': 0})

    counts = np.apply_over_axes(np.sum, edges, [0, 1]).ravel() 
    tots = np.prod(edges.shape[:2])

    ag.plot.images(np.rollaxis(edges, 2))

    return counts, tots

from multiprocessing import Pool
p = Pool(7)
res = map(process, files)

counts = sum(map(lambda x: x[0], res))
tots = sum(map(lambda x: x[1], res))

if 0:
    plt.hist(np.log(counts / tots)/np.log(10))
    plt.xlabel("log10(Features per pixel)")
    plt.ylabel("Frequency")
    plt.show()

print counts
print tots
print counts/tots

np.save('edge_bkg.npy', counts/tots)
