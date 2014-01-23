from __future__ import division
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
import numpy as np
import gv
import os
import glob
from scipy.stats import norm


eps = 1e-4
q_var = 0.001

def logpdf(X, mean, cov):
    """Minus constants"""
    diff = X - mean
    _, logabsdet = np.linalg.slogdet(cov)
    v = logabsdet + np.dot(diff, np.linalg.solve(cov, diff))
    return -v / 2

EPS = 0.01
def clog(x):
    return np.log(x.clip(EPS, 1-EPS))

def logtildegamma(G, Z, X, supp, w, G_mu, G_Sigma, eta):
    prob = gv.sigmoid(w + gv.logit(Z))
    llh = np.sum(supp * X * clog(prob)) + np.sum(supp * (1 - X) * clog(1 - prob))
    prior = logpdf(G, G_mu, G_Sigma)
    #print llh, prior
    return llh + prior

def find_valid(fun, l=-5, u=5, ol=-5, ou=5, depth=30):
    lf = fun(l)
    uf = fun(u)
    m = np.mean([l, u])
    if not np.isnan(lf) and not np.isnan(uf):
        return (l, u)

    if depth == 0:
        if np.isnan(lf):
            return (m, ou)
        if np.isnan(uf):
            return (ol, m)
    
    mf = fun(m)
    if np.isnan(lf):
        if np.isnan(mf):
            return find_nan(fun, m, u, ol, ou, depth-1)
        else:
            return find_nan(fun, l, m, ol, ou, depth-1)
        
    if np.isnan(uf):
        if np.isnan(mf):
            return find_nan(fun, l, m, ol, ou, depth-1)
        else:
            return find_nan(fun, m, u, ol, ou, depth-1)

def find_zero_inner(fun, l=-5, u=5, depth=30):
    m = np.mean([l, u])
    if depth == 0:
        return m
    
    v = fun(m)
    if v > 0:
        return find_zero_inner(fun, l, m, depth-1)
    else:
        return find_zero_inner(fun, m, u, depth-1)
        
def find_zero(fun, l=-5, u=5, depth=15):
    #l, u = find_valid(fun)
    return find_zero_inner(fun, l, u, depth)

def _do_sample(n, X, supp, Z, T, w, G_mu, G_Sigma, var, eta, seed):
    rs = np.random.RandomState(seed)
    F = Z.shape[0]
    G = np.log(Z)

    Gs_n = np.zeros((T, F))

    f_cur = logtildegamma(G, Z, X, supp, w, G_mu, G_Sigma, eta)
    acceptances = np.zeros(T)
    for t in xrange(T):
        G = np.log(Z)
        #G_mu = np.log(mu)

        #for i in xrange(1):
        new_G = rs.normal(loc=G, scale=var, size=F).clip(max=np.log(1 - EPS))
        new_Z = np.exp(new_G)

        #f_old = logtildegamma(Z, X, w, mu, Sigma)
        f_new = logtildegamma(new_G, new_Z, X, supp, w, G_mu, G_Sigma, eta)

        #mu[0]
        #Z[0]
        
        alpha = np.exp(f_new - f_cur)
        #print alpha
        accept = rs.uniform() < alpha
        if accept:
            Z = new_Z
            G = new_G
            f_cur = f_new
            #G = np.log(Z)

        Gs_n[t] = G
        acceptances[t] = accept
    return n, Gs_n, acceptances.mean()

if gv.parallel.main(__name__):
    #plot_dir = os.path.join(os.path.expandvars('$HOME'), 'html', 'plots')
    plot_dir = os.path.join(os.path.expandvars('$HOME'), 'plots')
    # Clear plotting directory
    for f in glob.glob(os.path.join(plot_dir, '*.png')):
        os.remove(f)

    d = gv.Detector.load('uiuc-np3b.npy')

    apos, aneg = [d.extra['sturf'][0][s] for s in ('pos', 'neg')]

    pos = apos[::10,...,::1]
    #F = d.num_features
    F = pos.shape[-1]

    Xbar = pos.mean(0).clip(EPS, 1-EPS)
    supp = d.extra['sturf'][0]['support'][...,np.newaxis]
    supp = np.tile(supp, (1, 1, F))

    # Initial parameters
    w = np.zeros(d.weights(0).shape[:2] + (F,))
    Z_mu = 0.1 * np.ones(F)
    G_mu = np.log(Z_mu)
    G_Sigma = 1.0 * np.eye(F)
    import time

    T = 8000
    burnin = 3000

    N = pos.shape[0]
    indices = None 
    indices_mask = None

    for loop in xrange(40):

        print 'Iteration {}'.format(loop+1)
        # Generate samples from posterior of Z
        start = time.time()
        Gs = np.zeros((N, T, F))

        #G_mu = np.zeros((N, F))
        #G_Sigma = np.zeros((N, F, F))

        if loop <= 2:
            var = 0.05
        elif loop <= 5:
            var = 0.025
        else:
            var = 0.005

        eta = min(1, 0.0 + 0.2 * loop)
            
        args = [(n, pos[n], supp, Z_mu.copy(), T, w, G_mu, G_Sigma, var, eta, loop*1234567+n) for n in xrange(N)]
        for n, Gs_n, acc_rate in gv.parallel.starmap_unordered(_do_sample, args):
            print 'Processed sample {} (acceptance rate: {})'.format(n, acc_rate)
            Gs[n] = Gs_n
        #for n in xrange(N): 
            #with gv.Timer('{loop} Sample {n}'.format(loop=loop+1, n=n+1)):

            #print 'Acceptance rate', acceptances.mean()

        print 'Samples collected'

        Gs_ok = Gs[:,burnin:].reshape((-1, F))
        Zs_ok = np.exp(Gs_ok)

        # Now use samples to estimate
        G_mu = Gs_ok.mean(0)
        Z_mu = np.exp(G_mu)

        #g_mu = np.exp(G_mu)
        G_Sigma = np.cov(Gs_ok.T) + np.eye(F) * 0.0001

        # Find w through a binary search   
        #for f in xrange(10):
        if 0:
            for f in xrange(F): 
                print 'Doing', f
                for l0, l1 in gv.multirange(*w.shape[:2]):
                    def fun(w0):
                        return gv.sigmoid(w0 + gv.logit(Zs_ok[...,f])).mean() - Xbar[l0,l1,f]

                    w[l0,l1,f] = find_zero(fun, l=-6, u=6, depth=15)

        if 0:
            l0 = l1 = 4
            f = 10

            def fun(w0):
                return gv.sigmoid(w0 + gv.logit(Zs_ok[...,f])).mean() - Xbar[l0,l1,f]

            #import IPython
            #IPython.embed()

            w[l0,l1,f] = find_zero(fun, l=-6, u=6, depth=15)


        LZ = gv.logit(Zs_ok.clip(EPS, 1-EPS))

        
        if 0 and loop == 0:
            import IPython
            IPython.embed()

        BINS = 50
        LZ_counts = np.zeros((F, BINS))
        LZ_values = np.zeros((F, BINS))
        for f in xrange(F):
            ret = np.histogram(LZ[...,f], BINS)
            LZ_counts[f] = ret[0]
            LZ_values[f] = (ret[1][1:] + ret[1][:-1])/2

        print LZ_counts[0]
        print LZ_values[0]

        print 'Finding zero...'
        from gv.fast import find_zeros_when_mcmc_training
        with gv.Timer('Finding zero'):
            w[:] = find_zeros_when_mcmc_training(Xbar, LZ_counts, LZ_values) 
        print 'w min/max', np.min(w), np.max(w)

        #if loop == 2:
        #    import IPython
        #    IPython.embed()

        if 1:
            # Print these to file
            from matplotlib.pylab import cm
            grid = gv.plot.ImageGrid(F, 1, w.shape[:2], border_color=(0.5, 0.5, 0.5))
            mm = 4#np.fabs(w).max()
            for f in xrange(F):
                grid.set_image(supp[...,f] * w[...,f], f, 0, vmin=-mm, vmax=mm, cmap=cm.RdBu_r)

            if loop == 0:
                np.save('w.npy', w)
            fn = os.path.join(plot_dir, 'plot{}.png'.format(loop))
            grid.save(fn, scale=4)
            os.chmod(fn, 0644)
        if 1:
            # Print these to file
            plt.figure()
            #plt.plot(np.exp(G_mu))
            G_std = np.sqrt(np.diag(G_Sigma))
            plt.errorbar(np.arange(F), np.exp(G_mu), yerr=[np.exp(G_mu) - np.exp(G_mu - G_std), np.exp(G_mu + G_std) - np.exp(G_mu)], fmt='o')
            #from matplotlib.pylab import cm
            #grid = gv.plot.ImageGrid(F, 1, w.shape[:2], border_color=(0.5, 0.5, 0.5))
            #mm = np.fabs(w).max()
            #for f in xrange(F):
                #grid.set_image(w[...,f], f, 0, vmin=-mm, vmax=mm, cmap=cm.RdBu_r)
            fn = os.path.join(plot_dir, 'plotb{}.png'.format(loop))
            #grid.save(fn, scale=6)
            plt.savefig(fn)
            os.chmod(fn, 0644)

        #np.linalg.solve(Sigma, 0.0 * np.ones(F))
        #plt.plot(new_Z)
        #end = time.time()
        #print end - start

        # Set indices
        if indices is None:
            from train_superimposed import get_key_points
            indices = get_key_points(w)

            indices_mask = np.zeros(w.shape, dtype=bool)
            for i, index in enumerate(indices):
                indices_mask[tuple(index)] = True
                #new_w[i] = w[tuple(index)]

            print 'supp shape', supp.shape
            print 'indices_mask shape', indices_mask.shape
            print 'indices shape', indices.shape
            print 'w shape', w.shape

            supp *= indices_mask
            
        w[~indices_mask] = 0

        # Update every iteration, so we can easily cancel
        np.savez('data.npz', w=w, G_mu=G_mu, G_Sigma=G_Sigma, indices=indices)

