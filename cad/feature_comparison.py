from __future__ import division
import glob
import os.path
import numpy as np
import amitgroup as ag
import pylab as plt

from config import SETTINGS

files = glob.glob(os.path.join(SETTINGS['src_dir'], '*.png'))

im = plt.imread(files[32])
imgrey = im[...,:3].mean(axis=2).astype(np.float64)

edges = ag.features.bedges(imgrey, radius=0)

edges2 = ag.features.bedges_from_image(im.astype(np.float64), radius=0)

num_edges, num_edges2 = edges.sum(), edges2.sum()
print "Edges (grayscale):", num_edges 
print "Edges (RGB):", num_edges2 

print "Increase:", num_edges2/num_edges

ag.plot.images(np.r_[edges, edges2])


