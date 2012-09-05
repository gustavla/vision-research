
from __future__ import division
import numpy as np
import amitgroup as ag
import matplotlib.pylab as plt
import scipy.special as sp
import sys

try:
    a0 = float(sys.argv[1])  
    b0 = float(sys.argv[2])
    N = int(sys.argv[3]) 
    precision = float(sys.argv[4])
except:
    print "<a0> <b0> <N> <precision>"
    sys.exit(0)

def make_gamma(a, b):
    @np.vectorize
    def _f(lmb):
        return 1/sp.gamma(a) * b**a * lmb**(a-1) * np.exp(-b*lmb)
    return _f

def caption(a, b):
    return "a0 = {0}, b0 = {1}, max = {2}, s.d. = {3}".format(a, b, (a-1)/b, np.sqrt(a/b**2))

x = np.linspace(0, 20, 200)

caption0 = caption(a0, b0)

aN = a0 + N/2
bN = b0 + N/2 / precision 

captionN = caption(aN, bN) 

print caption0
print captionN
PLOT = False

if PLOT:
    gamma0 = make_gamma(a0, b0)
    gammaN = make_gamma(aN, bN)

    plt.plot(gamma0(x), label=caption0)
    plt.plot(gammaN(x), label=captionN)
    plt.legend()
    plt.show()
