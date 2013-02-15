from __future__ import division
import numpy as np
import scipy.misc
import amitgroup as ag

def resize_to_size(im, new_size):
    return scipy.misc.imresize((im*255).astype(np.uint8), new_size).astype(np.float64)/255

def raw_resize(im, factor):
    new_size = tuple([int(round(im.shape[i] * factor)) for i in xrange(2)])
    # TODO: Change to something much more suited for this.
    return resize_to_size(im, new_size)

def _filtered_resize_once(im, new_size, preserve_aspect_ratio=True, prefilter=True):
    size = im.shape[:2]
    factor = new_size[0]/size[0]
    im2 = im.copy()
    if prefilter:
        im2 = ag.util.blur_image(im2, 1.5)
    im2 = raw_resize(im2, factor)
    return im2


def resize_with_factor(im, factor, preserve_aspect_ratio=True, prefilter=True):
    new_size = [int(im.shape[i] * factor) for i in xrange(2)]
    return resize(im, new_size, preserve_aspect_ratio=preserve_aspect_ratio, prefilter=prefilter)

import scipy.misc
def resize(im, new_size, preserve_aspect_ratio=True, prefilter=True):
    """
    This is a not-very-rigorous function to do image resizing, with
    pre-filtering.
    """
    factors = [new_size[i] / im.shape[i] for i in xrange(2)]

    if max(factors) > 1.0:
        # Just do resize
        return resize_to_size(im, new_size)
    else:
        im2 = im.copy()
        size = im.shape
        while max(factors) < 0.5:
            half_size = tuple([size[i]/2.0 for i in xrange(2)])
            factors = [factors[i] * 2.0 for i in xrange(2)]
            im2 = _filtered_resize_once(im2, half_size, preserve_aspect_ratio, prefilter)

            size = half_size

        # Now do the final one
        # TODO: Add filtering here as well
        if size != new_size:
            im2 = resize_to_size(im2, new_size)
    return im2

#from PIL import Image
import skimage.data
import os.path

def load_image(path):
    #im = np.array(Image.open(path))
    im = skimage.data.load(path)
    return im.astype(np.float64)/255.0
    #_, ext = os.path.splitext(path)
    #if ext.lower() in ['.jpg', '.jpeg']:
        #return im#[::-1]
    #else:
        #return im

def save_image(im, path):
    pil_im = Image.fromarray((im*255).astype(np.uint8))
    pil_im.save(path)

