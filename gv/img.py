from __future__ import division
import numpy as np
import scipy.misc

def resize(im, factor):
    new_size = tuple([int(round(im.shape[i] * factor)) for i in xrange(2)])
    # TODO: Change to something much more suited for this.
    return scipy.misc.imresize((im*255).astype(np.uint8), new_size).astype(np.float64)/255

from PIL import Image
import os.path

def load_image(path):
    im = np.array(Image.open(path))
    return im
    #_, ext = os.path.splitext(path)
    #if ext.lower() in ['.jpg', '.jpeg']:
        #return im#[::-1]
    #else:
        #return im
    
