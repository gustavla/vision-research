from __future__ import division
import numpy as np

def apply_filter(im, imfilter):
    if imfilter is None or imfilter == 'none':
        return im
    elif imfilter.endswith('-noise'):
        sigma = dict(low=0.01, mid=0.05, high=0.1)[imfilter.split('-')[0]]
        rs = np.random.RandomState(0)
        return (im + rs.normal(0, sigma, size=im.shape)).clip(0, 1)
    else:
        raise ValueError("Unknown filter name: {}".format(imfilter))


