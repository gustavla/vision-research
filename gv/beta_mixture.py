
from __future__ import division
import numpy as np
from scipy.stats import beta
from scipy.misc import logsumexp
import amitgroup as ag
import warnings

def binary_search(f, lower, upper, tol=0.001, maxiter=10):
    x0 = (lower+upper) / 2
    f0 = f(x0)
    if np.fabs(f0) < tol:
        return x0
    elif f0 > 0:
        return binary_search(f, lower, x0, tol=tol, maxiter=maxiter-1)
    else:
        return binary_search(f, x0, upper, tol=tol, maxiter=maxiter-1)

def weighted_avg_and_var(values, weights):
    """
    Returns the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.dot(weights, (values-average)**2)/weights.sum()  # Fast and numerically precise
    return (average, variance)

class BetaMixture(object):
    def __init__(self, n_clusters=2):
        self._n_clusters = n_clusters

        self.cluster_centers_ = None
        self.labels_ = None
        self.theta_ = None
        
    def fit(self, X, tol=0.00001, min_probability=0.01, min_q=0.01):
        Xsafe = np.clip(X, min_probability, 1 - min_probability)
        self._init(X, seed=0)
        M = self._n_clusters
        N = X.shape[0]
        D = X.shape[1]


        qlogs = np.zeros((M, N))
        q = np.zeros((M, N))
        v = np.zeros(N)

        loglikelihood = -np.inf
        new_loglikelihood = self._compute_loglikelihoods(Xsafe)
         
        self.iterations = 0
        while np.isinf(loglikelihood) or self.iterations < 2 or np.fabs((new_loglikelihood - loglikelihood)/loglikelihood) > tol:
            loglikelihood = new_loglikelihood
            ag.info("Iteration {0}: loglikelihood {1}".format(self.iterations, loglikelihood))
            for m in xrange(M):
                v = qlogs[m] 
                v[:] = 0.0
                for d in xrange(D):
                    #print beta.logpdf(Xsafe[:,d], self.theta_[m,d,0], self.theta_[m,d,1])
                    v += beta.logpdf(Xsafe[:,d], self.theta_[m,d,0], self.theta_[m,d,1])
                qlogs[m] = v
                #print v.min(), v.max()
                
            #try:
            try:
                q[:] = np.exp(np.maximum(np.log(min_q), qlogs - logsumexp(qlogs, axis=0)))
            except:
                pass
            #except:
            #    pass
            # Clip it, for saftey
            #print q.min(), q.max()
            q[:] = np.clip(q, min_q, 1 - min_q)

            # Update labels from these responsibilities
            self.labels_ = q.argmax(axis=0) 
            
            # Update thetas with the new labels
            if 0:
                for m in xrange(M):
                    for d in xrange(D):
                        Xsafem = Xsafe[self.labels_ == m, d]
                        sm, sv = weighted_avg_and_var(Xsafe[:,d], q[m])
                        #sm = np.mean(Xsafem)
                        #sv = np.var(Xsafem)
                        self.theta_[m,d,0] = sm * (sm * (1 - sm) / sv - 1)
                        self.theta_[m,d,1] = (1 - sm) * (sm * (1 - sm) / sv - 1)
                        if np.isnan(self.theta_[m,d,0]) or np.isnan(self.theta_[m,d,1]):
                            import pdb; pdb.set_trace()
                            raise Exception()

            else:
                for m in xrange(M):
                    for d in xrange(D):
                        #from scipy.optimize import newton_krylov, nonlin
                        from scipy.special import psi

                        Ca = np.average(np.log(Xsafe[:,d]), weights=q[m])
                        Cb = np.average(np.log(1-Xsafe[:,d]), weights=q[m])
                        a, b = self.theta_[m,d]

                            #self.theta_[m,d,0] = newton_krylov(lambda x: (psi(x) - psi(x+b)) - Ca, 1.0)
        
                        self.theta_[m,d,0] = binary_search(lambda x: (psi(x) - psi(x+b)) - Ca, 0.0001, 10000.0, maxiter=20)
    

                        self.theta_[m,d,1] = binary_search(lambda x: (psi(x) - psi(x+a)) - Cb, 0.0001, 10000.0, maxiter=20)
                        #C = np.average(    

            #self.theta_ = np.clip(self.theta_, 0.1, 100.0)
        
            # Calculate log-likelihood
            new_loglikelihood = self._compute_loglikelihoods(Xsafe)
            self.iterations += 1

        ag.info("Iteration DONE: loglikelihood {}".format(new_loglikelihood))

    def _compute_loglikelihoods(self, X):
        llh = 0.0
        M = self._n_clusters
        D = X.shape[1]
        #print self.theta_.min(), self.theta_.max()
        for m in xrange(M):
            for d in xrange(D):
                #print self.theta_[m,d,0], self.theta_[m,d,1]
                #print X[:,d].min(), X[:,d].max(), self.theta_[m,d]
                llh += np.sum((self.labels_ == m) * \
                    beta.logpdf(X[:,d], self.theta_[m,d,0], self.theta_[m,d,1]))
        
        #print X.min(), X.max()
        #print self.theta_.min(), self.theta_.max()
        #print 'LLH:::::::', llh

        return llh

    def _init(self, X, seed=0):
        prnd = np.random.RandomState(seed)
        M = self._n_clusters
        self.labels_ = prnd.randint(M, size=X.shape[0])
        D = X.shape[1]

        self.theta_ = np.zeros((M, D, 2))

        for m in xrange(M):
            Xm = X[self.labels_ == m]
            for d in xrange(D):
                sm = np.mean(Xm[:,d])
                sv = np.var(Xm[:,d])
                self.theta_[m,d,0] = sm * (sm * (1 - sm) / sv - 1)
                self.theta_[m,d,1] = (1 - sm) * (sm * (1 - sm) / sv - 1)

