
import numpy as np
import matplotlib.pylab as plt
import amitgroup as ag
import amitgroup.features

def find_patches(patches, image_file):
    #info = patch_data['info'].flat[0]

    edges, img = ag.features.bedges_from_image(image_file, k=5, radius=0, minimum_contrast=0.05, contrast_insensitive=True, return_original=True, lastaxis=True)

    # Now, pre-process the log parts
    log_parts = np.log(patches)
    log_invparts = np.log(1-patches)

    threshold = 4 

    ret = ag.features.code_parts(edges, log_parts, log_invparts, threshold)
    ret2 = ret.argmax(axis=2)

    K = 100#info['K'] 
    spread = ag.features.spread_patches(ret2, 3, 3, K)

    #print "spread:", spread.shape
    return ret2, spread, img

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

    ret2, spread, img = find_patches(patches, image_file)

    if 1:
        plt.subplot(121)
        plt.imshow(img, interpolation='nearest')
        plt.subplot(122)
        plt.imshow(ret2, interpolation='nearest')
        plt.colorbar()
        plt.show()
