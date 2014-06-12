
import numpy as np
import amitgroup as ag

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
        return tuple(self.lower[i] + (self.upper[i]-1 - self.lower[i]) * indices[i] / (self.shape[i]-1) for i in range(len(indices)))
        
    def ipos(self, indices):
        """
        Similar to :func:`pos`, except the position as a tuple of integral indices.
        """
        return tuple([int(x+0.0005) for x in self.pos(indices)])

    @classmethod
    def inner_frame(cls, orig_feat, padwidth):
        if not isinstance(padwidth, tuple):
            padwidth = (padwidth,) * orig_feat.ndim 

        new_lower = orig_feat.pos(padwidth)
        new_upper = orig_feat.pos(tuple(orig_feat.shape[i]-1-padwidth[i] for i in range(len(padwidth))))

        return new_lower, new_upper 
    
    @classmethod
    def zeropad(cls, feat, padwidth):
        if not isinstance(padwidth, tuple):
            padwidth = (padwidth,) * feat.ndim 

        padded_arr = ag.util.zeropad(feat, padwidth)
        new_lower = tuple(feat.lower[i] - (feat.upper[i] - 1 - feat.lower[i]) * padwidth[i] / (feat.shape[i] - 1) for i in range(len(feat.lower)))
        new_upper = tuple((feat.upper[i] - 1 - feat.lower[i]) / ((feat.shape[i] - 1) / (padded_arr.shape[i] - 1)) + 1 + new_lower[i] for i in range(len(feat.lower)))
        return ndfeature(padded_arr, lower=new_lower, upper=new_upper) 

if __name__ == '__main__':
    x = ndfeature(np.ones((50, 50)), lower=(0, 0), upper=(50, 50))
    print(x.pos((0, 0)))
    print(x.pos((49, 49)))
    assert x.pos((0, 0)) == (0, 0)
    assert x.pos((49, 49)) == (49, 49)
    print('OK')
    print('----')

    y = ndfeature(np.ones((50, 50, 10)), lower=(10, 10), upper=(90, 90))
    print(y.lower, y.upper)
    print(y.pos((0, 0)))
    print(y.pos((49, 49)))
    print(y.pos((1, 3)))

    print('----')
    z = ndfeature_zeropad(y, (10, 10, 0))
    print(z.lower, z.upper)
    print(z.ipos((10, 10)))
    print(z.ipos((49+10, 49+10)))
    print(z.ipos((1+10, 3+10)))
    
    print('----')
    #w = ndfeature_inflate_frame(ndfeature(z[10:-10,10:-10], lower=z.lower, upper=z.upper), (-10, -10))
    l, u = ndfeature.inner_frame(z, (10, 10))
    w = ndfeature(z[10:-10, 10:-10], lower=l, upper=u)
    print(w.lower, w.upper)
    print(w.ipos((0, 0)))
    print(w.ipos((49, 49)))
    print(w.ipos((1, 3)))
