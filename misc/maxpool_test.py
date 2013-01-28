from __future__ import division
import numpy as np
import amitgroup as ag
from time import time

def pool(data, length, func):
    steps = data.shape[1]//length
    ret = np.empty((data.shape[0], steps))
    for i in xrange(steps):
        ret[:,i] = func(data[:,i*length:(i+1)*length])
    return ret
    
def maxpool(data, length):
    return pool(data, length, lambda x: np.max(x, axis=1))

def meanpool(data, length):
    return pool(data, length, lambda x: np.mean(x, axis=1))

def minpool(data, length):
    return pool(data, length, lambda x: np.min(x, axis=1))

def trial(seed=0):
    eps = 1e-10
    start = time()
    ag.set_verbose(True)
    np.set_printoptions(precision=2, suppress=True)
    K = 2
    eachN = 100000 
    N = K*eachN
    M = 40 
    pool_length = 2 
    np.random.seed(0)
    theta = np.random.random((K, M))
    alphas = (np.random.random((K, M)) < 0.5).astype(float)
    print alphas.mean()
    #alphas = np.tile(np.random.random(M) < 0.5, (K, 1))
    #b = np.clip(np.random.random(M), 0.1, 0.9)
    b = np.ones(M) * 0.2
    b2 = np.ones(M//pool_length) * 0.36
    print theta
    print b

    np.random.seed(seed+1000)
    X = np.zeros((N, M))
    A = np.zeros((N, M))
    for k in xrange(K):
        for i in xrange(eachN):
            X[k*eachN+i] = np.random.random(M) < theta[k]
            A[k*eachN+i] = alphas[k]

    #X = (np.random.random((N, M)) < 0.5).astype(float)
    X *= A
    end = time() 
    print end - start
    Y = maxpool(X, pool_length)
    model = ag.stats.BernoulliMixture(K, Y.astype(np.uint8))
    model.run_EM(eps, 1e-5)
    B = meanpool(A, pool_length)
    B2 = maxpool(A, pool_length)
    print "#"*80
    print A.shape
    print B.shape
    support = model.remix(B)
    support2 = model.remix(B2)
    #corrected = model.templates + (1-(support+support2)/2) * b2
    corrected = 1 - (1-model.templates) * (1-b[0])**(pool_length*(1-support))
    print "X"
    print X
    print "alphas"
    print A
    #print "b"
    #print b

    N = K*eachN
    X2 = np.zeros((N, M))
    A2 = np.zeros((N, M))
    for k in xrange(K):
        for i in xrange(eachN):
            back = (np.random.random(M) < b).astype(float)
            fore = (np.random.random(M) < theta[k]).astype(float)
            X2[k*eachN+i] = alphas[k] * fore + (1-alphas[k]) * back

    print '---------->'
    print X2.shape
    Y2 = maxpool(X2, pool_length)
    print Y2.shape
    model2 = ag.stats.BernoulliMixture(K, Y2.astype(np.uint8))
    model2.run_EM(eps, 1e-5)
    #print X
    #print X2

    aff1 = model.affinities
    aff2 = model2.affinities
    print 's', aff1.shape

    print 'support'
    print support


    model2templates = np.clip(model2.templates, 0.05, 0.95)
    corrected = np.clip(corrected, 0.05, 0.95)
    model1templates = np.clip(model.templates, 0.05, 0.95)
    both = np.fabs([model2templates - corrected, model2templates[::-1] - corrected]).sum(axis=1).sum(axis=1)
    if np.argmin(both) == 1:
        model2templates = model2templates[::-1]
        aff2 = aff2[:,::-1]

    #print aff1
    #print aff2
        
    print 'model'
    print model1templates
    print 'corrected'
    print corrected
    print 'model2'
    print model2templates
    print '---'
    print (corrected - model2templates)
    scores = np.fabs(corrected - model2templates).sum(), np.fabs(model1templates - model2templates).sum()
    print scores[0], scores[1]

    return corrected, model2templates, scores

if __name__ == '__main__':
    TRIALS = 1
    for i in xrange(TRIALS):
        c, m, s = trial(i)
        #print c
        #print m
        #print s 
