# Benchmark of daubechies wavelet transformations 
import numpy as np
import pywt
import amitgroup as ag

db4g = np.array([0.6830127, 1.1830127, 0.3169873, -0.1830127]) / np.sqrt(2)
db4g = np.array([0.48296291314469025, 0.83651630373746899, 0.22414386804185735, -0.12940952255092145])
db4h = db4g[::-1].copy()
db4h[1::2] *= -1
N = len(db4g)


def _populate(W, filtr, yoffset):
    N = len(filtr)
    for i in range(W.shape[1]//2):
        for j in range(N):
            W[yoffset+i, (-(N-2)//2+2*i+j)%W.shape[1]] += filtr[j]


    if 0:
        fr, to = -1+2*i, -1+len(filtr)+2*i
        fr0, to0 = max(0, fr), min(W.shape[0], to)
        if fr != fr0:
            W[yoffset+i,fr:] = filtr[:-fr]
        if to != to0:
            W[yoffset+i,:to-W.shape[0]] = filtr[-(to-W.shape[0]):]
        W[yoffset+i,fr0:to0] = filtr[fr0-fr:fr0-fr+(to0-fr0)]

def _create_W(shape, level, filter_low, filter_high):
    d = 2**(level-1)
    W = np.zeros((shape[0]//d, shape[1]//d))
    _populate(W, filter_low, 0)
    _populate(W, filter_high, shape[0]//(2*d))
    return W

#W = np.zeros((32, 32))
#populate(W, db4g, 0)
#populate(W, db4h, 16)
sh = (32, 32)

W = _create_W(sh, 1, db4g, db4h)
W2 = _create_W(sh, 2, db4g, db4h)
W3 = _create_W(sh, 3, db4g, db4h)
W4 = _create_W(sh, 4, db4g, db4h)
W5 = _create_W(sh, 5, db4g, db4h)

G1 = np.zeros((16, 32))
G2 = np.zeros((8, 16))

_populate(G1, db4g, 0)
_populate(G2, db4g, 0)

print W[:len(W)//2,:]
print ';;;;;;;;;;;;;;'
print G1

np.testing.assert_array_almost_equal(W[:len(W)//2,:], G1)

#W4 = np.zeros((1, 1))
#populate(W3, db4g

print '----'
print W3
print '----'

if 0:
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

def qdot(X, A):
    return np.dot(X, np.dot(A, X.T))

def top_left_quad(A):
    N = len(A)//2
    return A[:N,:N]
def top_right_quad(A):
    N = len(A)//2
    return A[:N,N:]
def bottom_right_quad(A):
    N = len(A)//2
    return A[N:,N:]
def bottom_left_quad(A):
    N = len(A)//2
    return A[N:,:N]

A = np.arange(32*32).reshape((32, 32))
u = pywt.wavedec2(A, 'db2', mode='per', level=5)

WG = np.dot(W3, np.dot(G2, G1))
Ab = qdot(WG, A) 
B = top_left_quad(Ab)
Bb = qdot(W4, B)
C = top_left_quad(Bb)
Cc = qdot(W5, C)
D = top_left_quad(Cc)

imdef = ag.util.DisplacementFieldWavelet((32, 32), 'db2')

res = np.empty(64)

res[:4] = Cc.T.flatten()
res[4:8] = bottom_left_quad(Bb).flatten() 
res[8:12] = top_right_quad(Bb).flatten() 
res[12:16] = bottom_right_quad(Bb).flatten() 

res[16:32] = bottom_left_quad(Ab).flatten()
res[32:48] = top_right_quad(Ab).flatten()
res[48:64] = bottom_right_quad(Ab).flatten()


# Final
res2 = ag.util.DisplacementFieldWavelet.pywt2array(u, imdef.levelshape, 3, 3)

print res[:16]
print res2[:16]

np.testing.assert_array_almost_equal(res, res2)

#E = qdot(W5, D)
print D 
print u[1]

print '----'
A = np.arange(4).reshape(2, 2)
print A
B = qdot(W5, A)
print B
C = qdot(W5.T, B)
print C


if 0:
    print u[-1][0]
    print u[-1][1]
    print u[-1][2]

    print '---'
    rep = np.dot(W, np.dot(A, W.T))
    Nrep = len(rep)//2
    print rep[Nrep:,:Nrep]
    print rep[:Nrep,Nrep:]
    print rep[Nrep:,Nrep:]

    np.testing.assert_array_almost_equal(u[-1][0], rep[Nrep:,:Nrep])
    np.testing.assert_array_almost_equal(u[-1][1], rep[:Nrep,Nrep:])
    np.testing.assert_array_almost_equal(u[-1][2], rep[Nrep:,Nrep:])

    print u[-2][0]
    print u[-2][1]
    print u[-2][2]

    print '----'
    LL = rep[:Nrep,:Nrep]
    LLb = np.dot(G1, np.dot(A, G1.T))
    np.testing.assert_array_almost_equal(LL, LLb)
    print "OOK!!!!"

    rep2 = np.dot(W2, np.dot(LL, W2.T))
    Nrep2 = len(rep2)//2
    print rep2[Nrep2:,:Nrep2]
    print rep2[:Nrep2,Nrep2:]
    print rep2[Nrep2:,Nrep2:]

    print '++++'
    print u[-3][0]
    print u[-3][1]
    print u[-3][2]
    print '----'
    LL = rep2[:Nrep2,:Nrep2]
    print LL.shape, 'hj'
    rep3 = np.dot(W3, np.dot(LL, W3.T))
    Nrep3 = len(rep3)//2

    # Another way to calculate rep3
    W3G21 = np.dot(W3, np.dot(G2, G1))
    rep3b = np.dot(W3G21, np.dot(A, W3G21.T))
    #print rep3.shape, rep3b.shape
    np.testing.assert_array_almost_equal(rep3, rep3b)

    print rep3[Nrep3:,:Nrep3]
    print rep3[:Nrep3,Nrep3:]
    print rep3[Nrep3:,Nrep3:]

    #print u[-4][0]
    #print u[-4][1]
    #print u[-4][2]
    #print rep3[:Nrep3,:Nrep3]

    print "OK"

