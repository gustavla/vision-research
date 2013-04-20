from __future__ import division
import numpy as np
import scipy.misc
import amitgroup as ag
from skimage.transform.pyramids import pyramid_reduce
import sys

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

def resize(im, new_size, preserve_aspect_ratio=True, prefilter=True):
    """
    This is a not-very-rigorous function to do image resizing, with
    pre-filtering.
    """
    factors = [new_size[i] / im.shape[i] for i in xrange(2)]

    assert factors[0] == factors[1], "Must have same factor for now"
    f = factors[0] 
    
    if f < 1:
        im2 = pyramid_reduce(im, downscale=1/f)
    elif f > 1:
        # Implement 
        assert 0, "Implement this using pyramid_expand"

    assert im2.shape[:2] == tuple(new_size), "{0} != {1}".format(im2.shape, new_size)
     
    return im2

def old_resize(im, new_size, preserve_aspect_ratio=True, prefilter=True):

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

def asgray(im):
    if im.ndim == 2:
        return im
    else:
        return im[...,:3].mean(axis=-1)

def crop(im, size):
    diff = [im.shape[index] - size[index] for index in (0, 1)]
    im2 = im[diff[0]//2:diff[0]//2 + size[0], diff[1]//2:diff[1]//2 + size[1]]
    return im2

#from PIL import Image
import skimage.io
import os.path

def load_image(path):
    #im = np.array(Image.open(path))
    im = skimage.io.imread(path)
    return im.astype(np.float64)/255.0
    #_, ext = os.path.splitext(path)
    #if ext.lower() in ['.jpg', '.jpeg']:
        #return im#[::-1]
    #else:
        #return im

def load_image_binarized_alpha(path, threshold=0.2):
    im = load_image(path)
    assert im.ndim == 3, "Assumes RGB or RGBA"
    if im.shape[2] == 3:
        return im, None
    else:
        alpha = (im[...,3] > threshold)
        # im uses premultiplied alphas, so I need to fix this

        eps = sys.float_info.epsilon
        imrgb = (im[...,:3]+eps)/(im[...,3:4]+eps)
        
        return imrgb * alpha.reshape(alpha.shape+(1,)), alpha

def save_image(im, path):
    from PIL import Image
    pil_im = Image.fromarray((im*255).astype(np.uint8))
    pil_im.save(path)

def integrate(ii, r0, c0, r1, c1):
    """Use an integral image to integrate over a given window.

    Parameters
    ----------
    ii : ndarray
    Integral image.
    r0, c0 : int
    Top-left corner of block to be summed.
    r1, c1 : int
    Bottom-right corner of block to be summed.

    Returns
    -------
    S : int
    Integral (sum) over the given window.

    """
    # This line is modified
    S = np.zeros(ii.shape[-1]) 

    S += ii[r1, c1]

    if (r0 - 1 >= 0) and (c0 - 1 >= 0):
        S += ii[r0 - 1, c0 - 1]

    if (r0 - 1 >= 0):
        S -= ii[r0 - 1, c1]

    if (c0 - 1 >= 0):
        S -= ii[r1, c0 - 1]

    return S

