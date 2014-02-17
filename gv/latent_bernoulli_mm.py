
import numpy as np
import itertools as itr
import amitgroup as ag
from scipy.special import logit
from scipy.misc import logsumexp

# TEMP
import vz

class LatentBernoulliMM(object):
    def __init__(self, n_components=1, permutations=1, n_iter=20, n_init=1, random_state=0, min_probability=0.05, thresh=1e-8):
        if not isinstance(random_state, np.random.RandomState):
            random_state = np.random.RandomState(random_state)

        self.random_state = random_state
        self.n_components = n_components
        if isinstance(permutations, (int, long)):
            # Cycle through them
            P = permutations
            self.permutations = np.zeros((P, P))
            for p1, p2 in itr.product(xrange(P), xrange(P)):
                self.permutations[p1,p2] = (p1 + p2) % P
        else:
            self.permutations = np.asarray(permutations)

        self.n_iter = n_iter
        self.n_init = n_init
        self.min_probability = min_probability
        self.thresh = thresh

        self.weights_ = None
        self.means_ = None

    def fit(self, X):
        N, P, F = X.shape
        assert P == len(self.permutations) 
        K = self.n_components
        eps = self.min_probability
        #theta = self.random_state.uniform(eps, 1 - eps, size=(K, P, F))

        all_pi = []# np.zeros((self.n_init, K, P))
        all_mu = []
        all_loglikelihoods = []

        #import scipy.sparse
        #X = scipy.sparse.csr_matrix(X)
    
        for trial in xrange(self.n_init):
            pi = np.ones((K, P)) / (K * P)

            # Initialize
            clusters = self.random_state.randint(K, size=N)
            theta = np.asarray([np.mean(X[clusters == k], axis=0) for k in xrange(K)])
            theta[:] = np.clip(theta, eps, 1 - eps)

            # TEMP
            #nice_theta = theta.reshape((theta.shape[0]*theta.shape[1], 6, 6, 8))
            #nice_theta = np.rollaxis(nice_theta, 3, 1)
            #vz.image_grid(nice_theta, vmin=0, vmax=1, cmap=vz.cm.RdBu_r, name='iter0', scale=4)

            calc_loglikelihood = True

            self.q = np.empty((N, K, P))
            logq = np.empty((N, K, P))
            loglikelihood = None
            for loop in xrange(self.n_iter):
                if calc_loglikelihood:
                    ag.info("Iteration ", loop+1, 'log-likelihood', loglikelihood)
                else:
                    ag.info("Iteration ", loop+1)
                #logq[:] = np.log(pi)[np.newaxis,:,np.newaxis]

                #for k, p in itr.product(xrange(K), xrange(P)):
                #    logq[:,k,p] = np.log(pi[k,p])
                logq[:] = np.log(pi[np.newaxis])

                for p in xrange(P):
                    for shift in xrange(P):
                        #p0_ = (p + shift)%P
                        p0 = self.permutations[shift,p]
                        #assert p0 == p0_, self.permutations
                        #import pdb; pdb.set_trace()
                        logq[:,:,p] += np.dot(X[:,p0], logit(theta[:,shift]).T) + np.log(1 - theta[:,shift]).sum(axis=1)[np.newaxis]

                #self.q[:] = np.exp(logq)
                #normq = self.q / np.apply_over_axes(np.sum, self.q, [1, 2])
                #self.q /= np.apply_over_axes(np.sum, self.q, [1, 2])
                #q2 = np.exp(logq - logsumexp(logq.reshape((-1, logq.shape[-1])), axis=0)[...,np.newaxis,np.newaxis])
                norm_logq = (logq - logsumexp(logq.reshape((logq.shape[0], -1)), axis=-1)[...,np.newaxis,np.newaxis])#.clip(min=-200)
                q2 = np.exp(norm_logq)
                self.q[:] = q2

                # Try regularizing a bit
                #self.q[:] = self.q.clip(min=1e-4)
                #self.q[:] /= np.apply_over_axes(np.sum, self.q, [1, 2])

                #norm_logq = logq - logsumexp(logq.reshape((logq.shape[0], -1)), axis=-1)[...,np.newaxis,np.newaxis]
                #self.q[:] = np.exp(norm_logq)

                #dens = np.apply_over_axes(np.sum, self.q, [0, 2])
                log_dens = logsumexp(np.rollaxis(norm_logq, 2, 1).reshape((-1, norm_logq.shape[1])), axis=0)[np.newaxis,:,np.newaxis]
                dens = np.exp(log_dens)

                for p in xrange(P):
                    v = 0 #np.dot(self.q[:,:,0].T, X[:,0]) + np.dot(self.q[:,:,1].T, X[:,1])
                    for shift in xrange(P):
                        #p0_ = (p + shift)%P
                        p0 = self.permutations[shift,p]
                        #assert p0 == p0_, self.permutations
                        v += np.dot(self.q[:,:,shift].T, X[:,p0])

                    theta[:,p,:] = v
                np.seterr(all='raise')
                #try:
                #vz.image_grid(self.q[:100], vmin=0, vmax=1, cmap=vz.cm.RdBu_r, name='plot')
                #import IPython; IPython.embed()
                #import pdb; pdb.set_trace()
                theta = theta / dens.ravel()[:,np.newaxis,np.newaxis]


                # TODO: Shape is hard coded here
                #nice_theta = theta.reshape((theta.shape[0]*theta.shape[1], 6, 6, 8))
                #nice_theta = np.rollaxis(nice_theta, 3, 1)
                #vz.image_grid(nice_theta, vmin=0, vmax=1, cmap=vz.cm.RdBu_r, name='iter{}'.format(loop+1), scale=4)


                #new parts
                #pi[:] = np.apply_over_axes(np.sum, self.q, [0, 2])[0,:,0] / N
                pi[:] = np.apply_over_axes(np.sum, self.q, [0])[0,:,:] / N
                pi[:] = np.clip(pi, 0.0001, 1 - 0.0001)

                # TODO: KEEP THIS?
                #pi[:] = np.ones(pi.shape) / pi.shape

                theta[:] = np.clip(theta, eps, 1 - eps)
                #pi = np.clip(pi, eps, 1 - eps)


                # Calculate log likelihood
                if calc_loglikelihood:
                    loglikelihood = logsumexp(logq)
                

            #self.weights_ = pi
            #self.means_ = theta
            all_pi.append(pi)
            all_mu.append(theta)
            all_loglikelihoods.append(loglikelihood)

        print all_loglikelihoods
        best_i = np.argmax(all_loglikelihoods)
        self.weights_ = all_pi[best_i]
        self.means_ = all_mu[best_i]


    def mixture_components(self):
        """
        Returns a list of which mixture component each data entry is associate with the most. 

        Returns
        -------
        components: list 
            A list of length `num_data`  where `components[i]` indicates which mixture index the `i`-th data entry belongs the most to (results should be degenerate).
        """
        return np.asarray([np.unravel_index(self.q[n].argmax(), self.q.shape[1:]) for n in xrange(self.q.shape[0])])
