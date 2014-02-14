from __future__ import division
import numpy as np
import scipy.misc
import amitgroup as ag
import sys
from skimage.transform.pyramids import pyramid_reduce, pyramid_expand

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

def resize_with_factor_new(im, factor):
    if factor < 1 - 1e-8:
        im2 = pyramid_reduce(im, downscale=1/factor)
    elif factor > 1 + 1e-8:
        im2 = pyramid_expand(im, upscale=factor)
    else:
        im2 = im.copy()
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

    #assert factors[0] == factors[1], "Must have same factor for now"
    f = factors[0] 
    
    if f < 1:
        im2 = pyramid_reduce(im, downscale=1/f)
    elif f > 1:
        im2 = pyramid_expand(im, upscale=f)
    else:
        im2 = im

    assert im2.shape[:2] == tuple(new_size), "{0} != {1} (original size: {2})".format(im2.shape, new_size, im.shape)
     
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

def crop_to_bounding_box(im, bb):
    im2 = im[bb[0]:bb[2], bb[1]:bb[3]]
    return im2

#from PIL import Image
import os.path

def load_image(path):
    import skimage.io
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

def save_image(path, im):
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

def composite(fg_img, bg_img, alpha):
    return fg_img * alpha + bg_img * (1 - alpha) 

def offset(img, off):
    sh = img.shape
    if sh == (0, 0):
        return img
    else:
        x = np.zeros(sh)
        x[max(off[0], 0):min(sh[0]+off[0], sh[0]), \
          max(off[1], 0):min(sh[1]+off[1], sh[1])] = \
            img[max(-off[0], 0):min(sh[0]-off[0], sh[0]), \
                max(-off[1], 0):min(sh[1]-off[1], sh[1])]
        return x

def bounding_box(alpha):
    """This returns a bounding box of the support for a given component"""
    assert alpha.ndim == 2

    # Take the bounding box of the support, with a certain threshold.
    #print("Using alpha", self.use_alpha, "support", self.support)
    supp_axs = [alpha.max(axis=1-i) for i in xrange(2)]

    th = 0.5 
    # Check first and last value of that threshold
    bb = [np.where(supp_axs[i] > th)[0][[0,-1]] for i in xrange(2)]

    # This bb looks like [(x0, x1), (y0, y1)], when we want it as (x0, y0, x1, y1)
    #psize = self.settings['subsample_size']
    #ret = (bb[0][0]/psize[0], bb[1][0]/psize[1], bb[0][1]/psize[0], bb[1][1]/psize[1])

    return (bb[0][0], bb[1][0], bb[0][1], bb[1][1])

def bounding_box_as_binary_map(alpha):
    bb = bounding_box(alpha)
    x = np.zeros(alpha.shape, dtype=np.uint8)
    x[bb[0]:bb[2], bb[1]:bb[3]] = 1
    return x 
