from __future__ import division
import numpy as np

def imshow(*args, **kwargs):
    import matplotlib.pylab as plt
    if 'interpolation' not in kwargs:
        kwargs['interpolation'] = 'nearest'
    return plt.imshow(*args, **kwargs)

def imshow_even(x, *args, **kwargs):
    import matplotlib.pylab as plt

    mm = np.fabs(x).max()
    if 'vminmax' in kwargs:
        mm = kwargs['vminmax']
    if 'interpolation' not in kwargs:
        kwargs['interpolation'] = 'nearest'
    if 'vmin' not in kwargs:
        kwargs['vmin'] = -mm
    if 'vmax' not in kwargs:
        kwargs['vmax'] = mm
    if 'cmap' not in kwargs:
        kwargs['cmap'] = plt.cm.RdBu_r

    return plt.imshow(x, *args, **kwargs)

def argmaxi(obj):
    return np.unravel_index(np.argmax(obj), obj.shape)

def bclip(x, eps):
    """Bernoulli clip. Clips `x` to the interval ``[eps, 1 - eps]``."""

    return np.clip(x, eps, 1 - eps)

def logit(x):
    return np.log(x / (1 - x))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def multirange(*args):
    import itertools
    return itertools.product(*map(xrange, args))


class SlicesClass(object):
    def __getitem__(self, *args):
        return args
slices = SlicesClass() 

import time
class Timer(object):
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        print "TIMER {0}: {1} s".format(self.name, self.end - self.start)
