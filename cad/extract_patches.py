
import argparse

parser = argparse.ArgumentParser(description='Train mixture model on edge data')
parser.add_argument('output', metavar='<output file>', type=argparse.FileType('wb'), help='Filename of patches file')

args = parser.parse_args()
output_file = args.output

import numpy as np
import amitgroup as ag
from config import SETTINGS
import glob
import os.path
import sys
import random

N = 100 
patch_size = 5,5
K = 25 

def num_patches_in_img(img_size, patch_size):
    return (img_size[0]-patch_size[0]+1)*(img_size[1]-patch_size[1]+1)

def gen_patches(img, edges, patch_size):
    for x in xrange(img.shape[0]-patch_size[0]+1):
        for y in xrange(img.shape[1]-patch_size[1]+1):
            selection = [slice(x, x+patch_size[0]), slice(y, y+patch_size[1])]
            # Return grayscale patch and edges patch
            yield img[selection], edges[selection]

#patches_per_image = num_patches_in_img(
patches = []
originals = []

files = glob.glob(os.path.join(SETTINGS['src_dir'], '*.png'))
random.seed(0)
random.shuffle(files)
for f in files[:N]:
    print "File", f
    edges, img = ag.features.bedges_from_image(f, k=5, radius=1, minimum_contrast=0.05, return_original=True, lastaxis=True)
    # Make grayscale
    imggray = img[...,:3].mean(axis=2)
    for impatch, edgepatch in gen_patches(imggray, edges, patch_size):
        if edgepatch[1:-1,1:-1].sum() >= 20:
            originals.append(impatch)
            patches.append(edgepatch)  
    
patches = np.asarray(patches)
originals = np.asarray(originals)

PLOT=False

#ag.plot.images(originals[:100])

mixture = ag.stats.BernoulliMixture(K, patches, init_seed=0)
mixture.run_EM(1e-4, min_probability=0.01, debug_plot=PLOT)

#orig_components = mixture.


#print "vispatches:", vispatches.shape
print "Originals:", originals.shape, originals.dtype
print "Shape:", patches.shape, patches.dtype

vispatches = mixture.remix(originals)

np.savez(output_file, patches=mixture.templates, originals=vispatches)

vispatches /= vispatches.max()

ag.plot.images(vispatches)

