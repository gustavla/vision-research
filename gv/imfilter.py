from __future__ import division
import numpy as np

def filter_names():
    return ['none', 'low-noise', 'mid-noise', 'high-noise', 'low-gamma', 'high-gamma']

def apply_filter(im, imfilter, seed=0):
    if imfilter is None or imfilter == 'none':
        return im
    elif imfilter.endswith('-noise'):
        sigma = dict(low=0.01, mid=0.05, high=0.1)[imfilter.split('-')[0]]
        rs = np.random.RandomState(seed)
        return (im + rs.normal(0, sigma, size=im.shape)).clip(0, 1)
    elif imfilter == 'random-gamma':
        rs = np.random.RandomState(seed)
        gamma = np.exp(rs.normal(loc=0, scale=1.0))
        return im**gamma
    elif imfilter.endswith('-gamma'):
        gamma = dict(low=0.5, high=2.0)[imfilter.split('-')[0]]
        return im**gamma
    else:
        raise ValueError("Unknown filter name: {}".format(imfilter))


