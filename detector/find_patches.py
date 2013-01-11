
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
    parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')
    parser.add_argument('img_id', metavar='<image id>', type=int, help='ID of image in VOC repository')
    parser.add_argument('--single-part', dest='part', nargs=1, default=[None], metavar='PART', type=float, help='Plot a single part')

    args = parser.parse_args()
    model_file = args.model
    img_id = args.img_id
    part_id = args.part[0]

    from config import VOCSETTINGS

    fileobj = gv.voc.load_training_file(VOCSETTINGS, 'bicycle', img_id)
    img = gv.img.load_image(fileobj.path)

    detector = gv.Detector.load(model_file)
    parts = detector.extract_features(img)
    #parts#= patch_dictionary.extract_parts_from_image(img, spread=False, return_original=True)

    #orientations = patch_orientations(detector.patch_dictionary.patches)
    #plt.hist(orientations, 30)
    #plt.show()

    #print orientations.shape
    #print orientations
    #sys.exit(0)
    
    if part_id is not None:
        #print parts.shape
        #plt.imshow(parts[...,part_id], interpolation='nearest')
        #plt.show()

        levels = parts.sum(axis=0).sum(axis=0) / np.prod(parts.shape[:2])
        print levels.mean()
        print parts.shape
        print levels.shape
        plt.plot(levels)
        plt.show()

    else:
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
