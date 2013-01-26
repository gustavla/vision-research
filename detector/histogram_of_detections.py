

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
        np.random.seed(0)
        originals, bbs = gv.voc.load_negative_images_of_size(VOCSETTINGS, 'bicycle', detector.kernel_size, count=50, padding=padding) 
    else:
        profiles = map(int, open('profiles.txt').readlines())
        originals, bbs = gv.voc.load_object_images_of_size_from_list(VOCSETTINGS, 'bicycle', detector.kernel_size, profiles, padding=padding) 

    print "NUMBER OF IMAGES", len(originals)

    limit = None 

    reses = []
    llhs = []
    # Extract features
    for i in xrange(len(originals)):
        im = originals[i]  
        grayscale_img = im.mean(axis=-1)
        bb = bbs[i]
        #edges = detector.extract_pooled_features(im)

        # Now remove the padding
        #edges = edges[padding:-padding,padding:-padding]

        #edgemaps.append(edges)
        #plt.imshow(im)
        #plt.show()        

        # Check response map
        print "calling response_map", im.shape, mixcomp
        res, small = detector.response_map(grayscale_img, mixcomp)

        print 'small', small.shape

        # Check max at the center of the bounding box (bb)
        ps = detector.settings['pooling_size']
        m = int((bb[0]+bb[2])/ps[0]//2), int((bb[1]+bb[3])/ps[1]//2)
        #m = res.shape[0]//2, res.shape[1]//2
        s = 2
        #print 'factor', self.factor(
        #print 'ps', ps
        #print 'im', im.shape
        #print 'res', res.shape
        #print m
        top = res[max(0, m[0]-s):min(m[0]+s, res.shape[0]), max(0, m[1]-s):min(m[1]+s, res.shape[1])].max()
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
                plt.imshow(im, interpolation='nearest')
                #plt.subplot(3, 6, 2+2*i)
                plt.subplot(1, 2, 2)
                plt.imshow(res, interpolation='nearest')
                plt.colorbar()
                #plt.title("{0}".format(i))
                plt.title("Top: {0:.2f} ({1:.2f})".format(top, res.max()))
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
