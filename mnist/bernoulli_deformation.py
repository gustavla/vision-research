from __future__ import division
import amitgroup as ag
import amitgroup.ml
import numpy as np
from time import time
from scipy.stats import norm


def main():
    if 0:
        def create_norm(pos, sigma):
            @np.vectorize
            def norm2(x, y):
                return norm.pdf(x, pos, sigma) * np.sqrt(sigma**2 * 2 * np.pi)
            return norm2

        norm2 = create_norm(32-5, 1.0)
        norm2b = create_norm(32-10, 1.0)

        x, y = np.mgrid[0:32, 0:32] 
        Fraw = norm2(x, y)
        F = ag.features.bedges(Fraw, k=6, inflate=True)
        F = np.rollaxis(F, axis=2).astype(float)
        Iraw = norm2b(x, y)
        I = ag.features.bedges(Iraw, k=6, inflate=True)
        I = np.rollaxis(I, axis=2).astype(float)
        for j in range(8):
            F[j] = ag.util.blur_image(F[j].astype(float), 4)
            if F[j].max() != 0:
                F[j] /= F[j].max() 
            #I[j] /= 3

        #I = ag.util.blur_image(I, 5)
        #ag.plot.images([Fraw, F[0], I[0]])
        #import sys; sys.exit(0)
            
        # Generate F and I
        pass
    elif 1:
        t1 = time()
        I = np.load('a-nine-features.npy')
        I = np.rollaxis(I, axis=2)
        Iraw = ag.util.zeropad(ag.io.load_example('mnist')[2], 2)
        I = ag.features.bedges(Iraw, k=5, inflate=True)
        I = np.rollaxis(I, axis=2)
        b = 0
        if b:
            Iraw = ag.util.blur_image(Iraw, b)
            I = ag.features.bedges(Iraw, k=5, inflate=True)
            I = np.rollaxis(I, axis=2)
            ag.plot.images([Iraw]+list(I))
        
        mixture_index = 5 
        F = np.load('a-mixtures-6-tighter.npz')['templates'][9,mixture_index]
        F = np.rollaxis(F, axis=2)
        #ag.plot.images(list(F) + [I[0]])
        #F = np.clip(F, 0.05, 0.95)
        #ag.plot.images((F >= 0.5).astype(float))
        print F.min(), F.max()
        print I.shape, F.shape


    if 0:
        imdef = ag.util.DisplacementFieldWavelet(F.shape[1:], 'db1')
        if 0:
            N = 1
            xx = [-0.0]
        else:
            N = 200 
            xx = np.linspace(-3, 10, N)
        llhs = np.empty(N) 
        del_llhs = np.empty(N)
        for i, u in enumerate(xx):
            imdef.u[0,1,0,0,0] = u 
            llh = ag.ml.loglikelihood(F, I, imdef)
            llhs[i] = llh
            del_llhs[i] = ag.ml.del_llh(F, I, imdef) * 10

            print 'i', i, 'u', u, 'likelihood', llh
            #ag.plot.images(imdef.deform(I[0]))

        #import sys; sys.exit(0)

        import matplotlib.pylab as plt
        if N > 1:
            grad = np.gradient(llhs, xx[1]-xx[0])
            #plt.plot(xx, grad * 1000, label='numerical del')
        plt.plot(xx, llhs, label='llhs')
        plt.plot(xx, del_llhs, label='del llhs')
        ii = np.argmax(llhs)
        print "max:", llhs[ii], " : x =", xx[ii]
        plt.legend()
        plt.show()
        import sys; sys.exit(0)

    imdef, info = ag.ml.bernoulli_deformation(F, I, stepsize_scale_factor=1.0, penalty=1.0, rho=2.0, last_level=4, tol=0.001, \
                                        start_level=1, wavelet='db8', max_iterations_per_level=2000, debug_plot=True)

    Favg = F.sum(axis=0)
    Fdef = imdef.deform(Favg)
    #t2 = time()

    #print "Time:", t2-t1

    PLOT = False
    if PLOT:
        import matplotlib.pylab as plt
        x, y = imdef.meshgrid()
        Ux, Uy = imdef.deform_map(x, y) 

        ag.plot.deformation(Favg, I[0], imdef)

        #Also print some info before showing
        print "Iterations (per level):", info['iterations_per_level'] 
    
        print imdef.u[0,1,0,0,0]
        
        # Print 
        #plt.semilogy(-info['loglikelihoods'])
        plt.plot(info['loglikelihoods'])
        plt.show()

if __name__ == '__main__':
    #import cProfile as profile
    #profile.run('main()')
    main()
