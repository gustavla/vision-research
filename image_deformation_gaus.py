from __future__ import division

import amitgroup as ag
import amitgroup.ml
import numpy as np
from time import time
from scipy.stats import norm



def main():
    x, y = np.mgrid[0:1.0-1/32.0:32j, 0:1.0-1/32.0:32j]
    N = 200
    xb, yb = np.mgrid[0:1.0-1/N:N*1j, 0:1]

    penalty = 0.1 / 32 

    if 0:
        @np.vectorize
        def norm2(x, y):
            sigma = 0.05
            return norm.pdf(x, 1/2+1/12, sigma) * np.sqrt(sigma**2 * 2 * np.pi)

        @np.vectorize
        def norm2b(x, y):
            sigma = 0.05
            x = x
            return norm.pdf(x, 1/2-1/12, sigma) * np.sqrt(sigma**2 * 2 * np.pi)

        F = norm2b(x, y)
        I = norm2(x, y) 


    else:
        means = []
        means2 = []
        priors = []

        shifts = np.linspace(0, 5.3333, 40)
        for shift in shifts:
            #shift = -float(sys.argv[1] )
            psi = lambda x: 1/32
            psi = np.vectorize(psi)

            if 1:
                @np.vectorize
                def norm2(x, y):
                    sigma = 0.05
                    return norm.pdf(x, 1/2-1/12, sigma) * np.sqrt(sigma**2 * 2 * np.pi)

                @np.vectorize
                def norm2b(x, y):
                    sigma = 0.05
                    x = x + shift * psi(x)# * 32
                    return norm.pdf(x, 1/2+1/12, sigma) * np.sqrt(sigma**2 * 2 * np.pi)

            else:
                @np.vectorize
                def norm2(x, y):
                    if 0 < x < 1/3:
                        return 3*x
                    elif 1/3 <= x <= 2/3:
                        return 2 - 3*x
                    else:
                        return 0.0

                @np.vectorize
                def norm2b(x, y):
                    x = x - x * shift * 1/6
                    if 1/3 <= x <= 2/3:
                        return 3*x - 1 
                    elif 2/3 <= x <= 1:
                        return 3 - 3*x 
                    else:
                        return 0.0

            import pylab as plt

            Fb = norm2b(xb, yb)
            Ib = norm2(xb, yb)
            oneI = Ib[:,0]
            oneF = Fb[:,0]
            delOneF = np.gradient(oneF, 1/N) 

            termsb = (oneF - oneI) * delOneF
            #m = termsb.sum() / N
            m2 = 0.5 * ((oneF - oneI)**2).sum() / N 
            m = (termsb * psi(xb[:,0])).sum() / N * 5.3333
            prior = penalty * shift**2
            #m = (oneF - oneI)
            #m2 = (termsb**2).sum() / N  / 2.0
            print shift, m2
            means.append(m)
            means2.append(m2)
            priors.append(prior)

        if 1:
            means = np.asarray(means)# * 5.3333# * 6
            means2 = np.asarray(means2)
            priors = np.asarray(priors)

            delm2 = np.gradient(np.asarray(means2), 1/40)

            plt.plot(shifts, means2, label='means2')
            plt.plot(shifts, means, label='means')
            plt.plot(shifts, delm2, label='delm2')
            plt.plot(shifts, priors, label='prior')
            plt.plot(shifts, priors+means2, label='neg. cost')
            plt.legend()
            print ':', delm2.max()
            #plt.plot(x0, -(0.33333-x0))
            #plt.plot(x0, np.asarray(means)/-(0.3333-x0))
            
            plt.show()

        if 0:
            plt.subplot(131)
            plt.plot(oneF)
            plt.subplot(132)
            plt.plot(delOneF)
            plt.subplot(133)
            plt.plot(termsb)
            plt.show()

        import sys; sys.exit(0)

    #ag.plot.images([F, I])
    #import pylab as plt
    #plt.show()

    t1 = time()

    #penalty /= 32 * 32
    imdef, info = ag.ml.imagedef(F, I, stepsize=None, penalty=penalty, rho=1.0, A=1, tol=0.0001, \
                                 max_iterations_per_level=50, start_level=1, calc_costs=True, wavelet='db1')
    Fdef = imdef.deform(F)
    t2 = time()

    print t2-t1

    if 1:
        ag.plot.deformation(F, I, imdef, show_diff=True)

        #Also print some info before showing
        print info['iterations_per_level'] 
        
        x, y = imdef.get_x(F.shape)
        Ux, Uy = imdef.deform_map(x, y)
        print 'max', Ux.max()
        print Ux[0,0] * 32, 'pixels'
        print (imdef.u[0,0,0,0,0]), 'pixels'
        #print imdef.u[1,0,0,0,0], -0.1
        #print imdef.u[0,1,0,0,0], 0.15
        #print imdef.u[0,1,1,0,0], -0.2
        #print imdef.u[1,1,2,0,0], 0.3

        # Print 
        import pylab as plt
        plt.semilogy(info['costs'])
        plt.show()

if __name__ == '__main__':
    #import cProfile as profile
    #profile.run('main()')
    main()
