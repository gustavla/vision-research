from __future__ import division
import numpy as np
import amitgroup as ag
from time import time


def trial(seed=0):
    start = time()
    ag.set_verbose(True)
    np.set_printoptions(precision=2, suppress=True)
    K = 2
    eachN = 100000 
    N = K*eachN
    M = 50 
    np.random.seed(0)
    theta = np.random.random((K, M))
    #alphas = (np.random.random((K, M)) < 0.5).astype(float)
    alphas = np.tile(np.random.random(M) < 0.5, (K, 1))
    #b = np.clip(np.random.random(M), 0.1, 0.9)
    b = np.ones(M) * 0.2
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
    model = ag.stats.BernoulliMixture(K, X.astype(np.uint8))
    model.run_EM(1e-6, 1e-5)
    support = model.remix(A)
    corrected = model.templates + (1-support) * b
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

    model2 = ag.stats.BernoulliMixture(K, X2.astype(np.uint8))
    model2.run_EM(1e-6, 1e-5)
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
    scores = np.fabs(corrected - model2templates).sum(), np.fabs(model1templates - model2templates).sum()
    print scores[0], scores[1]

    return corrected, model2templates, scores

if __name__ == '__main__':
    TRIALS = 1
    for i in xrange(TRIALS):
        c, m, s = trial(i)
        print c
        print m
        print s 
