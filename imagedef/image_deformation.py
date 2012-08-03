from __future__ import division

import amitgroup as ag
import amitgroup.ml
import numpy as np
from time import time
from scipy.stats import norm



def main():
    #x, y = np.mgrid[0:1.0-1/F.shape[0]:F.shape[0]*1j, 0:1.0-1/F.shape[1]:F.shape[1]*1j]
    wavelet = 'db2'
    if 0:
        F = ag.io.load_example('faces2')[0]
        imdef2 = ag.util.DisplacementFieldWavelet(F.shape, wavelet=wavelet)
        imdef2.u[0,0,0,0,0] = 2.1 
        if 1:
            imdef2.u[0,1,0,1,0] = 1.3
            imdef2.u[0,1,0,1,1] = -1.25
            imdef2.u[0,1,1,0,0] = -1.3
            imdef2.u[0,1,1,0,1] = 1.45
            
            imdef2.u[1,2,0,0,0] = 0.2
            imdef2.u[1,2,0,0,1] = -0.1
            imdef2.u[1,2,0,3,3] = 0.3
            imdef2.u[1,2,1,1,1] = 1.1 
        I = imdef2.deform(F)

        x, y = imdef2.meshgrid()
    elif 0:
        F, I = ag.io.load_example('faces2')
    else:
        F_, I_ = ag.io.load_example('mnist')[:2]
        F = np.zeros((32, 32))
        I = np.zeros((32, 32))
        F[2:-2,2:-2] = F_
        I[2:-2,2:-2] = I_
        
        N = 64 
        M = 64 
        #N = M = 32
        M = 32
        # Resample
        if N != F.shape[0]:
            from scipy.signal import resample
            F = resample(resample(F, N, axis=0), M, axis=1)
            I = resample(resample(I, N, axis=0), M, axis=1)

    penalty = 2.0 
    #dt = 0.01 * 32 * 32
    t1 = time()
    imdef, info = ag.ml.imagedef(F, I, penalty=penalty, rho=2.0, start_level=1, last_level=5,
                                 tol=0.001, stepsize_scale_factor=3.0, max_iterations_per_level=1000, calc_costs=True, \
                                 wavelet=wavelet)
    t2 = time()
    print "Time", (t2 -t1)
    print "Iterations", info['iterations_per_level']
    Fdef = imdef.deform(F)

    if 1:
        ag.plot.deformation(F, I, imdef, show_diff=False)

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

        #print 'u0', imdef.u[0,0,0,0,0], imdef2.u[0,0,0,0,0]
    
        # Print 
        #import pylab as plt
        #plt.semilogy(info['costs'])
        #plt.show()

if __name__ == '__main__':
    #import cProfile as profile
    #profile.run('main()')
    main()
