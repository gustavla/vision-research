
import numpy as np
import itertools as itr
import amitgroup as ag
from scipy.special import logit

class LatentBernoulliMM(object):
    def __init__(self, n_components=1, n_latents=1, n_iter=20, random_state=0, min_probability=0.05, thresh=1e-8):
        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)

        self.random_state = random_state
        self.n_components = n_components
        self.n_latents = n_latents
        self.n_iter = n_iter
        self.min_probability = min_probability
        self.thresh = thresh

        self.weights_ = None
        self.means_ = None

    def fit(self, X):
        N, P, F = X.shape
        assert P == self.n_latents
        K = self.n_components
        eps = self.min_probability
        pi = np.ones((K, P)) / (K * P)
        #theta = self.random_state.uniform(eps, 1 - eps, size=(K, P, F))

        # Initialize
        clusters = np.random.randint(K, size=N)
        theta = np.asarray([np.mean(X[clusters == k], axis=0) for k in xrange(K)])
        theta[:] = np.clip(theta, eps, 1 - eps)

        self.q = np.empty((N, K, P))
        logq = np.empty((N, K, P))
        for i in xrange(self.n_iter):
            ag.info("Iteration ", i+1)
            #logq[:] = np.log(pi)[np.newaxis,:,np.newaxis]

            #for k, p in itr.product(xrange(K), xrange(P)):
            #    logq[:,k,p] = np.log(pi[k,p])
            logq[:] = np.log(pi[np.newaxis])

            for p in xrange(P):
                #p0, p1 = p, (p+1)%2
                for shift in xrange(P):
                    p0 = (p+shift)%P
                    logq[:,:,p] += np.dot(X[:,p0], logit(theta[:,shift]).T) + np.log(1 - theta[:,shift]).sum(axis=1)[np.newaxis]

            import scipy.misc
#
            #self.q[:] = np.exp(logq)
            #normq = self.q / np.apply_over_axes(np.sum, self.q, [1, 2])
            #self.q /= np.apply_over_axes(np.sum, self.q, [1, 2])
            #q2 = np.exp(logq - scipy.misc.logsumexp(logq.reshape((-1, logq.shape[-1])), axis=0)[...,np.newaxis,np.newaxis])
            norm_logq = logq - scipy.misc.logsumexp(logq.reshape((logq.shape[0], -1)), axis=-1)[...,np.newaxis,np.newaxis]
            q2 = np.exp(norm_logq)
            self.q[:] = q2

            #norm_logq = logq - scipy.misc.logsumexp(logq.reshape((logq.shape[0], -1)), axis=-1)[...,np.newaxis,np.newaxis]
            #self.q[:] = np.exp(norm_logq)

            #dens = np.apply_over_axes(np.sum, self.q, [0, 2])
            log_dens = scipy.misc.logsumexp(np.rollaxis(norm_logq, 2, 1).reshape((-1, norm_logq.shape[1])), axis=0)[np.newaxis,:,np.newaxis]
            dens = np.exp(log_dens)

            for p in xrange(P):
                v = 0 #np.dot(self.q[:,:,0].T, X[:,0]) + np.dot(self.q[:,:,1].T, X[:,1])
                for shift in xrange(P):
                    v += np.dot(self.q[:,:,shift].T, X[:,(p+shift)%P])

                theta[:,p,:] = v
            theta /= dens.flatten()[:,np.newaxis,np.newaxis]

            #new parts
            #pi[:] = np.apply_over_axes(np.sum, self.q, [0, 2])[0,:,0] / N
            pi[:] = np.apply_over_axes(np.sum, self.q, [0])[0,:,:] / N
            pi[:] = np.clip(pi, 0.0001, 1 - 0.0001)

            # TODO: KEEP THIS?
            #pi[:] = np.ones(pi.shape) / pi.shape

            theta[:] = np.clip(theta, eps, 1 - eps)
            #pi = np.clip(pi, eps, 1 - eps)

        self.weights_ = pi
        self.means_ = theta

    def mixture_components(self):
        """
        Returns a list of which mixture component each data entry is associate with the most. 

        Returns
        -------
        components: list 
            A list of length `num_data`  where `components[i]` indicates which mixture index the `i`-th data entry belongs the most to (results should be degenerate).
        """
        return np.asarray([np.unravel_index(self.q[n].argmax(), self.q.shape[1:]) for n in xrange(self.q.shape[0])])
