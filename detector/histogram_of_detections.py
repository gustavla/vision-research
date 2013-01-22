

# Let's pad a little bit, so that we get the features correctly at the edges

import argparse
import matplotlib.pylab as plt
import amitgroup as ag
import gv
import numpy as np
from config import VOCSETTINGS

def main():

    parser = argparse.ArgumentParser(description='Train mixture model on edge data')
    parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of the model file')
    parser.add_argument('mixcomp', metavar='<mixture component>', type=int, help='mix comp')
    parser.add_argument('--negatives', action='store_true', help='Analyze n')

    args = parser.parse_args()
    model_file = args.model
    mixcomp = args.mixcomp
    negatives = args.negatives

    # Load detector
    detector = gv.Detector.load(model_file)

    llhs = calc_llhs(VOCSETTINGS, detector, not negatives, mixcomp)

    plt.hist(llhs, 10)
    plt.show()

def calc_llhs(VOCSETTINGS, detector, positives, mixcomp):
    padding = 0 
    if not positives:
        images = gv.voc.load_negative_images_of_size(VOCSETTINGS, 'bicycle', detector.kernel_size, count=300, padding=padding) 
    else:
        profiles = map(int, open('profiles.txt').readlines())
        images = gv.voc.load_object_images_of_size_from_list(VOCSETTINGS, 'bicycle', detector.kernel_size, profiles, padding=padding) 

    print "NUMBER OF IMAGES", len(images)

    limit = None 

    reses = []
    llhs = []
    # Extract features
    for i, im in enumerate(images[:limit]):
        #edges = detector.extract_pooled_features(im)

        # Now remove the padding
        #edges = edges[padding:-padding,padding:-padding]

        #edgemaps.append(edges)

        # Check response map
        res, small = detector.response_map(im, mixcomp)

        # Check max
        m = res.shape[0]//2, res.shape[1]//2
        s = 3
        top = res[m[0]-s:m[0]+s, m[1]-s:m[1]+s].max()
        llhs.append(top)

        if 1:
            if limit is not None:
                plt.subplot(3, 6, 1+2*i)
                plt.imshow(im, interpolation='nearest')
                plt.subplot(3, 6, 2+2*i)
                plt.imshow(res, interpolation='nearest')
                plt.colorbar()
                plt.title("Top: {0:.2f} ({1:.2f})".format(top, res.max()))
            elif False:#top < -5000:
                #plt.subplot(3, 6, 1+2*i)
                plt.subplot(1, 2, 1)
                plt.imshow(images[i], interpolation='nearest')
                #plt.subplot(3, 6, 2+2*i)
                plt.subplot(1, 2, 2)
                plt.imshow(res, interpolation='nearest')
                plt.colorbar()
                plt.title("{0}".format(i))
                #plt.title("Top: {0:.2f} ({1:.2f})".format(top, res.max()))
                plt.show()
        
    #print llhs
    if 0:
        if limit is not None:
            plt.show()
        else:
            plt.hist(llhs, 10)
            plt.show()

    return np.asarray(llhs)

if __name__ == '__main__':
    main() 
