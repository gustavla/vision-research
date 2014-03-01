
import numpy as np

import gv
import time
import itertools

def f(x, y):
    time.sleep(1)
    return 2*y + x

def f_star(args):
    return f(*args)

N = 30

if gv.parallel.main(__name__):
    x = np.arange(N)
    y = 4 * np.arange(N)
    args = [(x[i], y[i]) for i in xrange(N)]
    for z in gv.parallel.starmap(f, args):
        print z
        
