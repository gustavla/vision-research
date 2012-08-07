from __future__ import division

import amitgroup as ag
import amitgroup.ml
import numpy as np
from time import time
from scipy.stats import norm



def main():
    x, y = np.mgrid[0:1.0-1/32.0:32j, 0:1.0-1/32.0:32j]
    N = 200
    #xb, yb = np.mgrid[0:1.0-1/N:N*1j, 0:1.0-1/N:N*1j]
    xb, yb = np.mgrid[0:1.0-1/N:N*1j, 0:1]

    #theta = 1.0
    penalty = 0.2 / 32 

    if 1:
        @np.vectorize
        def norm2(x, y):
            sigma = 0.05
            return norm.pdf(x, 1/2-1/12, sigma) * np.sqrt(sigma**2 * 2 * np.pi)

        @np.vectorize
        def norm2b(x, y):
            sigma = 0.05
            return norm.pdf(x, 1/2+1/12, sigma) * np.sqrt(sigma**2 * 2 * np.pi)

        F = norm2b(x, y)
        I = norm2(x, y) 


    else:
        means = []
        means2 = []
        priors = []
        #delFs = []
        #del2Fs = []
        A = []
        B = []

        shifts = np.linspace(0, 5.3333, 40)
        #shifts = [3.528]
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

            F = norm2b(x, y)
            I = norm2(x, y) 
            delF = np.gradient(F, 1/len(x), 1/len(x))

            termsb = (oneF - oneI) * delOneF
            terms = (F - I) * delF
            print terms.shape

            #delFs.append(delOneF)
            delOne2F = np.gradient(delOneF, 1/len(x))
            #del2Fs.append(delOne2F)
            A.append((delOne2F * (oneF - oneI)).mean())
            B.append((delOneF**2).mean())
            

            #import pywt
            #u = pywt.wavedec2(terms[0], 'db1', mode='per') 
            #print u[0]
            #import sys; sys.exit(0)

            #m = termsb.sum() / N
            m2 = -0.5 * ((oneF - oneI)**2).sum() / N 
            m = -(termsb * psi(xb[:,0])).sum() / N# * 5.3333
            prior = 0.5 * -penalty * shift**2
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

            delm2 = np.gradient(means2, shifts[-1]/40)
            delpriors = np.gradient(priors, shifts[-1]/40)

            del2m2 = np.gradient(delm2, shifts[-1]/40)
            del2priors = np.gradient(delpriors, shifts[-1]/40)
            costs = -(priors+means2)
            delcosts = np.gradient(costs, shifts[-1]/40)
            del2costs = np.gradient(delcosts, shifts[-1]/40)

            plt.plot(shifts, means2, label='means2')
            #plt.plot(shifts, means, label='means')
            plt.plot(shifts, delm2, label='del llh')
            plt.plot(shifts, del2m2, label='del2 llh')
            plt.plot(shifts, priors, label='prior')
            plt.plot(shifts, delpriors, label='del priors')
            plt.plot(shifts, del2priors, label='del2 priors')
            plt.plot(shifts, delcosts, label='del costs')
            plt.plot(shifts, del2costs, label='del2 costs')
            #plt.plot(shifts, del2, label='del2 costs')
            plt.plot(shifts, costs, label='cost')
            plt.plot(shifts, A, label='A')
            plt.plot(shifts, B, label='B')
            #plt.plot(shifts, delFs, label='del F')
            #plt.plot(shifts, del2Fs, label='del2 F')
            plt.legend()
            print ':', delm2.max()
            print '--->', shifts[np.argmin(-(priors+means2))]
            print np.min(-(priors+means2))
            #plt.plot(x0, -(0.33333-x0))
            #plt.plot(x0, np.asarray(means)/-(0.3333-x0))

            #print (priors+means2)
            
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
    #dt = 0.01 * 32 * 32
    imdef, info = ag.ml.imagedef(F, I, stepsize_scale_factor=0.5, penalty=penalty, rho=1.0, last_level=3, tol=0.0001, \
                                 max_iterations_per_level=500, start_level=1, wavelet='db1')
    Fdef = imdef.deform(F)
    t2 = time()

    print t2-t1

    if 1:
        ag.plot.deformation(F, I, imdef, show_diff=True)

        #Also print some info before showing
        print info['iterations_per_level'] 
        
        x, y = imdef.meshgrid()
        Ux, Uy = imdef.deform_map(x, y)
        print 'max', Ux.max()
        print Ux[0,0] * 32, 'pixels'
        #print (imdef.u[0,0,0,0,0]), 'pixels'
        print "u[0] = ", imdef.u[0,0,0,0,0]
        #print imdef.u[1,0,0,0,0], -0.1
        #print imdef.u[0,1,0,0,0], 0.15
        #print imdef.u[0,1,1,0,0], -0.2
        #print imdef.u[1,1,2,0,0], 0.3
    
        import pylab as plt
        plt.plot(info['logpriors'], label='logpriors')
        plt.plot(info['loglikelihoods'], label='llh')
        plt.plot(info['costs'], label='costs') 
        plt.legend()
        plt.show()

        # Print 
        #import pylab as plt
        #plt.semilogy(info['costs'])
        #plt.show()

if __name__ == '__main__':
    #import cProfile as profile
    #profile.run('main()')
    main()
