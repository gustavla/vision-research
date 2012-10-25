
import numpy as np
import amitgroup as ag
from config import SETTINGS
import glob
import os.path
import sys

N = 10 
patch_size = 5,5
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

for f in glob.glob(os.path.join(SETTINGS['src_dir'], '*.png'))[:N]:
    patches = []
    edges, img = ag.features.bedges_from_image(f, return_original=True, lastaxis=True)    
    # Make grayscale
    imggray = img[...,:3].mean(axis=2)
    for impatch, edgepatch in gen_patches(imggray, edges, patch_size):
        if edgepatch.sum() > 0:
            originals.append(impatch)
            patches.append(edgepatch)  
    
patches = np.asarray(patches)
originals = np.asarray(originals)

K = 20

PLOT=False

mixture = ag.stats.BernoulliMixture(K, patches, init_seed=0)
mixture.run_EM(1e-10, min_probability=0.05, debug_plot=PLOT)

#orig_components = mixture.


print "Originals:", originals.shape, originals.dtype
print "Shape:", patches.shape, patches.dtype

ag.plot.images(originals[100:125])

