
import numpy as np
import amitgroup as ag
from config import SETTINGS
import glob
import os.path
import sys
import random

from patchmodel import random_patches
from find_patches import find_patches
from do_blocks import make_block_matrix

def filter_patches(percentile, block, patches, vispatches):
    counts = block.sum(axis=0) 

    import pylab as plt
    plt.hist(counts, 30)
    plt.show()

    import scipy.stats
    cutoff = scipy.stats.scoreatpercentile(counts, percentile)

    keep_indices = np.where(counts < cutoff)

    plt.hist(counts[keep_indices], 30)
    plt.show()


    patches = patches[keep_indices]
    vispatches = vispatches[keep_indices]
    return patches, vispatches

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train mixture model on edge data')
    parser.add_argument('output', metavar='<patches file>', type=argparse.FileType('wb'), help='Filename of patches file')
    parser.add_argument('--filter', action='store_true', help='Filter the most common patches')

    args = parser.parse_args()
    output_file = args.output


    N = 10
    patch_size = (5, 5)
    big_patch_frame = 1 
    K = 200 

    ag.set_verbose(True)

    files = glob.glob(os.path.join(SETTINGS['patches_dir'], '*'))
    random.seed(0)
    random.shuffle(files)
    validate_files = files[-50:]
    raw_patches, raw_originals, num_edges = random_patches(files[:20], patch_size, samples_per_image=500)

    #import matplotlib.pylab as plt
    #plt.hist(num_edges, 30)
    #plt.show()

    #print raw_originals.shape
    # Make grayscale
    #raw_originals = raw_originals[...,:3].mean(axis=raw_originals.ndim-1)

    ag.info("Training...")
    mixture = ag.stats.BernoulliMixture(K, raw_patches, init_seed=0)
    mixture.run_EM(1e-4, min_probability=0.01, debug_plot=False)
    ag.info("Done.")
    import pdb; pdb.set_trace()

    # Store the stuff in the instance
    patches = mixture.templates

    mix0 = mixture.indices_lists()[0]


    info = {
        'K': K,
        'patch_size': patch_size,
    }

    vispatches = mixture.remix(raw_originals)

    np.savez('patches-old.npz', patches=patches, vispatches=vispatches, info=info)


    if args.filter:
        blocks = []
        for f in validate_files:
            feat, spread, img = find_patches(patches, f)
            block = make_block_matrix(feat)
            blocks.append(block)

        b = np.hstack(blocks)

        #print b.shape
        patches, vispatches = filter_patches(80, b, patches, vispatches)
            #print "Counting in", f
        #def patch_metric(patch):
        #    return np.median(patch)

        #scores = []
        #for patch in patches:
        #    scores.append( patch_metric(patches) )
        #print scores
        
        #for patch in patches:
            #if 

    ag.plot.images(vispatches)

    np.savez(output_file, patches=patches, vispatches=vispatches, info=info)

