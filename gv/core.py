import numpy as np

def imshow(*args, **kwargs):
    import matplotlib.pylab as plt
    if 'interpolation' not in kwargs:
        kwargs['interpolation'] = 'nearest'
    return plt.imshow(*args, **kwargs)

def argmaxi(obj):
    return np.unravel_index(np.argmax(obj), obj.shape)

