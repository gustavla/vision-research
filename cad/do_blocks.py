from __future__ import division
import numpy as np
from find_patches import find_patches, find_patch_logprobs


def make_block_matrix(data):
    K = 200
    blocksize = (5, 5)
    size = data.shape[:2]

    blocks = tuple([size[i]//blocksize[i] for i in xrange(2)])
    num_blocks = np.prod(blocks)
    mat = np.zeros((K, num_blocks))
    block_i = 0
    for i in xrange(blocks[0]):
        for j in xrange(blocks[1]):
            x, y = i*blocksize[0], j*blocksize[1]
            bl = data[x:x+blocksize[0], y:y+blocksize[1]]
            d = bl.flatten()
            for index in d:
                if index > 0:
                    #print block_i, index
                    mat[index-1,block_i] += 1
            block_i += 1
    return mat

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train mixture model on edge data')
    parser.add_argument('patch', metavar='<patches file>', type=argparse.FileType('rb'), help='Filename of patches file')
    parser.add_argument('image', metavar='<image file>', type=argparse.FileType('rb'), help='Filename of image file to detect in')

    args = parser.parse_args()
    patch_file = args.patch
    image_file = args.image

    patch_data = np.load(patch_file)
    patches = patch_data['patches']
    #ret2, spread, img = find_patches(patches, image_file)
    ret, img = find_patch_logprobs(patches, image_file)
    ret2 = ret.argmax(axis=-1).flatten()
    num_non_background = np.prod(ret2[ret2 != 0].size)
    
    # Display a single patch
    print ret.shape
    import pylab as plt
    
    # Find the most ubiquitous patch class
    maxes = ret.argmax(axis=-1).flatten()
    best = np.bincount(maxes)
    most_common = np.argmax(best[1:])+1
    num_most_common = ret2[ret2 == most_common].size
    print "Most common patch: {0} ({1:.2f}% prevalence of non-background)".format(most_common, 100*num_most_common/num_non_background)
    
    
    plt.hist(maxes, 200)
    plt.show()

    
    # Calculate the mean log probs for each probability, and then plot that as a histogram
    K = 200
    means = np.zeros(K)
    for i in xrange(1, K+1):
        x = ret[...,i].flatten()
        x = x[x != -np.inf]
        means[i-1] = x.mean()
        
    plt.hist(means, 30)
    plt.title("Mean log probabilities of all patches in bikes.png")
    plt.show()
    
    if 1:
        for i in xrange(9):
            plt.subplot(3, 3, 1+i)
            x = ret[...,85+i].flatten()
            x = x[x != -np.inf]
            plt.hist(x, 30)
            plt.title("Patch {0}".format(85+i))
        
    plt.show()
    
    import sys; sys.exit(0)
    
    ret2 = ret.argmax(axis=-1)
    #print ret2.shape

    import matplotlib.pylab as plt
    mat = make_block_matrix(ret2)
    cor = np.dot(mat, mat.T)

    m = -np.inf
    mi = None
    cors = []
    for x in xrange(cor.shape[0]):
        for y in xrange(x+1, cor.shape[1]):
            cors.append( (cor[x,y], (x,y)))
            if cor[x,y] > m:
                m = cor[x,y]
                mi = x,y

    cors = sorted(cors)[::-1]
    print "argmax =", mi

    import amitgroup as ag
    #ag.plot.images(np.rollaxis(patches[mi[0]], axis=2))
    #ag.plot.images(np.rollaxis(patches[mi[1]], axis=2))
    vispatch = patch_data['vispatches']

    # Filter some
    #from train_patches import filter_patches
    #patches, vispatches = filter_patches(90, cors, patches, vispatches)


    if 0:
        for c, mi in cors[:10]:
            print mi, c
            ag.plot.images([vispatch[mi[0]]] + list(np.rollaxis(patches[mi[0]], axis=2)) + [vispatch[mi[1]]] + list(np.rollaxis(patches[mi[1]], axis=2)))
            import sys; sys.exit(0)


    u, s, v = np.linalg.svd(mat)
    plt.plot(s)
    plt.show()
    plt.imshow(u)
    plt.colorbar()
    plt.show()

    if 1:
        plt.imshow(mat, interpolation='nearest')
    else:
        plt.plot(mat.mean(axis=0))
    plt.show()
