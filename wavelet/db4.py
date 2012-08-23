# Benchmark of daubechies wavelet transformations 
import numpy as np
import pywt

db4g = np.array([0.6830127, 1.1830127, 0.3169873, -0.1830127]) / np.sqrt(2)
db4g = np.array([0.48296291314469025, 0.83651630373746899, 0.22414386804185735, -0.12940952255092145])
db4h = db4g[::-1].copy()
db4h[1::2] *= -1
N = len(db4g)


def populate(W, filtr, yoffset):
    N = len(filtr)
    for i in range(W.shape[1]//2):
        for j in range(N):
            W[yoffset+i, (-(N-2)//2+2*i+j)%W.shape[0]] += filtr[j]


    if 0:
        fr, to = -1+2*i, -1+len(filtr)+2*i
        fr0, to0 = max(0, fr), min(W.shape[0], to)
        if fr != fr0:
            W[yoffset+i,fr:] = filtr[:-fr]
        if to != to0:
            W[yoffset+i,:to-W.shape[0]] = filtr[-(to-W.shape[0]):]
        W[yoffset+i,fr0:to0] = filtr[fr0-fr:fr0-fr+(to0-fr0)]

W = np.zeros((8, 8))
populate(W, db4g, 0)
populate(W, db4h, 4)

W2 = np.zeros((4, 4))
populate(W2, db4g, 0)
populate(W2, db4h, 2)

W3 = np.zeros((2, 2))
populate(W3, db4g, 0)
populate(W3, db4h, 1)

#W4 = np.zeros((1, 1))
#populate(W3, db4g

print '----'
print W3
print '----'

print pywt.wavedec(np.arange(8), 'db2', mode='per', level=3)
ret = np.dot(W, np.arange(8)) #/ np.sqrt(2)
print ret

ret2 = np.dot(W2, ret[:len(ret)//2]) #/ np.sqrt(2)
print ret2

ret3 = np.dot(W3, ret2[:len(ret2)//2]) #/ np.sqrt(2)
print ret3

#ret4 = np.dot(W4, ret3[:len(ret3)//2] / np.sqrt(2)
#print ret4

print '+'*80

u = pywt.wavedec2(np.arange(64).reshape(8, 8), 'db2', mode='per', level=3)
print u[-1][0]
print u[-1][1]
print u[-1][2]

print '---'
rep = np.dot(W, np.dot(np.arange(64).reshape((8, 8)), W.T))
Nrep = len(rep)//2
print rep[Nrep:,:Nrep]
print rep[:Nrep,Nrep:]
print rep[Nrep:,Nrep:]

np.testing.assert_array_almost_equal(u[-1][0], rep[Nrep:,:Nrep])
np.testing.assert_array_almost_equal(u[-1][1], rep[:Nrep,Nrep:])
np.testing.assert_array_almost_equal(u[-1][2], rep[Nrep:,Nrep:])

LL = rep[:Nrep,:Nrep]
#rep2 = np.dot(W2, np.dot(

print "OK"
