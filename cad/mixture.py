
from config import SETTINGS


import glob
import os.path

import numpy as np
import amitgroup as ag

def mixture_from_files(files, K):
    images = [] 
    originals = []

    for f in files:
        im = edges, original = ag.features.bedges_from_image(f, k=5, radius=1, return_original=True, contrast_insensitive=False, lastaxis=True)
        images.append(edges)
        originals.append(original)
        
    images = np.asarray(images)
    originals = np.asarray(originals)

    print originals.shape

    mixture = ag.stats.BernoulliMixture(K, images, init_seed=0)
    mixture.run_EM(1e-4)

    return mixture, images, originals 


if __name__ == '__main__':
    files = glob.glob(os.path.join(SETTINGS['src_dir'], "*.png"))
    mixture, images, originals = mixture_from_files(files, 6)
    vismix = mixture.remix(originals[...,:3].mean(axis=originals.ndim-1))
    ag.plot.images(vismix)

