
import numpy as np
import matplotlib.pylab as plt
import amitgroup as ag
import amitgroup.features
import scipy.linalg as LA
import math
import gv

def patch_orientations(patches):
    N = len(patches)
    orientations = np.empty(N)
    for i in xrange(N):
        total_v = np.array([0.0, 0.0]) 
        for e in xrange(8):
            angle = e*(2*np.pi)/8
            v = np.array([np.cos(angle), np.sin(angle)])
            count = patches[i,...,e].sum()
            total_v += v * count 
            
        if total_v[0] != 0 or total_v[1] != 0:
            total_v /= LA.norm(total_v)    
        orientations[i] = math.atan2(total_v[1], total_v[0])
    return orientations
         
        

def find_patches(patches, image_file):
    ret, img = find_patch_logprobs(patches, image_file)
    ret2 = ret.argmax(axis=2)

    K = 100#info['K'] 
    spread = ag.features.spread_patches(ret2, 3, 3, K)

    #print "spread:", spread.shape
    return ret2, spread, img

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Find and visualize patches in an image')
    parser.add_argument('patches', metavar='<patches file>', type=argparse.FileType('rb'), help='Filename of patches file')
    parser.add_argument('image', metavar='<image file>', type=argparse.FileType('rb'), help='Filename of image file to detect in')

    args = parser.parse_args()
    patch_file = args.patches
    image_file = args.image

    patch_dictionary = gv.PatchDictionary.load(patch_file)
    parts, img = patch_dictionary.extract_parts_from_image(image_file, spread=False, return_original=True)

    orientations = patch_orientations(patch_dictionary.patches)
    #plt.hist(orientations, 30)
    #plt.show()

    #print orientations.shape
    #print orientations
    #sys.exit(0)

    if 1:
        ors = np.empty(parts.shape)
        for i in xrange(ors.shape[0]):
            for j in xrange(ors.shape[1]):
                ors[i,j] = orientations[parts[i,j]-1]#%np.pi

    if 1:
        plt.subplot(121)
        plt.imshow(img, interpolation='nearest')
        plt.subplot(122)
        #plt.imshow(parts, interpolation='nearest', cmap=plt.cm.hsv)
        plt.imshow(ors, interpolation='nearest', cmap=plt.cm.hsv)
        plt.colorbar()
        plt.show()
