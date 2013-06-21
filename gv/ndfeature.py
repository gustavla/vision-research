from __future__ import division
import numpy as np

class ndfeature(np.ndarray):

    def __new__(cls, input_array, lower=None, upper=None):
        obj = np.asarray(input_array).view(cls)
        # Position of lower index
        obj.lower = lower
        # Position of one pixel above upper index
        obj.upper = upper
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.info = getattr(obj, 'info', None)

    def pos(self, indices):
        """
        Given an index tuple, returns the position in the original image corresponding
        to that index. 
        """
        return tuple(self.lower[i] + (self.upper[i]-1 - self.lower[i]) * indices[i] / (self.shape[i]-1) for i in xrange(len(indices)))
        
    def ipos(self, indices):
        """
        Similar to :func:`pos`, except the position as a tuple of integral indices.
        """
        return tuple(map(int, self.ipos(indices)))



if __name__ == '__main__':
    x = ndfeature(np.ones((50, 50)), lower=(0, 0), upper=(50, 50))
    print x.pos((0, 0))
    print x.pos((49, 49))
    assert x.pos((0, 0)) == (0, 0)
    assert x.pos((49, 49)) == (49, 49)
    print 'OK'

    y = ndfeature(np.ones((50, 50, 10)), lower=(10, 10), upper=(91, 91))
    print y.pos((0, 0))
    print y.pos((49, 49))
    print y.pos((1, 3))
