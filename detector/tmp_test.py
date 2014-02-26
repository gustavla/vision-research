
import numpy as np

import gv

def f(x, y):
    return 2*y + x

def starf(args):
    return f(*args)

N = 30

if gv.parallel.main(__name__):
    x = np.arange(N)
    y = 4 * np.arange(N)
    args = [(x[i], y[i]) for i in xrange(N)]
    #for i, z in enumerate(gv.parallel.starmap(f, args)):
    for i, z in enumerate(gv.parallel.imap(starf, args)):
        print i, ':', 2*y[i] + x[i], z
        assert 2*y[i] + x[i] == z
        
