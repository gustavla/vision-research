
import numpy as np
from scipy import interpolate
from scipy import linalg
import amitgroup as ag
from copy import copy

twopi = 2.0 * np.pi

def lerp(a, x, y):
    return y * a + (1-a) * x 

def interp2d(x, y, _z, fill_value=0.0):
    z = copy(_z)
 #   @np.vectorize
    def interp2d_f(p):
        for i in range(len(x[:,0])-1):
            for j in range(len(y[0,:])-1):
                if (x[i+1,0] > p[0] >= x[i,0] and y[0,j+1] > p[1] >= y[0,j]):
                    a = (p[0]-x[i,0])/(x[i+1,0]-x[i,0])
                    xp1 = lerp(a, z[i,j], z[i+1,j])
                    xp2 = lerp(a, z[i,j+1], z[i+1,j+1])
                    a = (p[1]-y[0,j])/(y[0,j+1]-y[0,j])
                    return lerp(a, xp1, xp2)
        return fill_value 
    return interp2d_f

from math import cos

def psi(k1, k2, x):
    #return twopi * np.real(np.exp(twopi * 1j * (k1 * x[0] + k2 * x[1])))
    return twopi * cos(twopi * (k1 * x[0] + k2 * x[1]))

def scriptN(a):
    return a

def inv_Psi(uxi, a):
    def inv_Psi_f(x):
        t = 0.0
        for k1 in xrange(scriptN(a)):
            for k2 in xrange(scriptN(a)):
                #print k1, k2
                t += uxi[k1,k2] * psi(k1, k2, x)
        return t 
    return inv_Psi_f 

def imagedef(F, I):
    """
    F: Prototype
    I: Image that will be deformed
    """
    xs = np.empty(F.shape + (2,))
    for x0 in range(F.shape[0]):
        for x1 in range(F.shape[1]):
            xs[x0,x1] = np.array([float(x0)/(F.shape[0]), float(x1)/F.shape[1]])

    xs0 = xs[:,:,0]
    xs1 = xs[:,:,1]

    px = xs1
    py = xs0

    # Calculate the gradients of F 
    # (the gradient does not know dx from just F, so we must multiply by dx)
    delF = np.gradient(F)
    delF[0] /= F.shape[0]
    delF[1] /= F.shape[0]
    ix, iy = np.mgrid[0:F.shape[0]-1:1j*F.shape[0], 0:F.shape[1]-1:1j*F.shape[1]]
    #_interpF = ag.math.interp2d(0, F, fill_value=0.0)
    #def interpF(p):
    #    return _interpF(np.array([p[0]*F.shape[0], p[1]*F.shape[1]]))

    
    interpF = ag.math.interp2d_func(F, fill_value=0.0)
    interpF.dx = 1.0/np.array(F.shape)
    #_interpF = interp2d(ix, iy, F)
    #tck = interpolate.bisplrep(ix, iy, F, s=0)
    #def interpF(x):
        #return _interpF(x[0]*F.shape[0], x[1]*F.shape[1])[0]
        #return _interpF(x[0]*F.shape[0], x[1]*F.shape[1])[0]
    #    return 0.0

    def fI(x):
        x, y = int(round(x[0]*F.shape[0])), int(round(x[1]*F.shape[1])) 
        return I[x,y]

    # Test stuff
    if 0:
        import sys
        from random import uniform 
        q = 0
        
        _f = ag.math.interp2d_func(delF[q], fill_value=0.0)
        f1 = lambda p:_f(np.array([p[0]*F.shape[0], p[1]*F.shape[1]]))
        f2 = interp2d(q, xs[:,:,0], xs[:,:,1], delF[q])

        for i in range(10000):
            x = np.array([uniform(0.0,0.9), uniform(0.0,0.9)])
            assert np.fabs(f1(x)-f2(x))<0.0001, "Not: {0}".format(x) 


        sys.exit(0)

    interpDelF = []
    #interpDelF2 = []
    for q in range(len(delF)):
        #_f = ag.math.interp2d(q, delF[q], fill_value=0.0)
        #_f.dx = 
        #interpDelF_f = lambda q, p:_f(q, np.array([p[0]*F.shape[0], p[1]*F.shape[1]]))
        interpDelF_f = ag.math.interp2d_func(delF[q], fill_value=0.0)
        interpDelF_f.startx = np.zeros(2)
        interpDelF_f.dx = 1.0/np.array(F.shape)
        #interpDelF_f2 = interp2d(xs[:,:,0], xs[:,:,1], delF[q])
            
        interpDelF.append(interpDelF_f)
        #interpDelF2.append(interpDelF_f2)

    #import sys; sys.exit(0)
    
    if 0:
        #xnew, ynew = np.mgrid[0:F.shape[0]-1:1j*F.shape[0],0:F.shape[1]-1:1j*F.shape[1]]
        xnew, ynew = np.mgrid[0:F.shape[0]-1:1j*F.shape[0]*4-1,0:F.shape[1]-1:1j*F.shape[1]*4-1]
        #xnew, ynew = np.mgrid[0:F.shape[0]-1:1j*F.shape[0]/2.0,0:F.shape[1]-1:1j*F.shape[1]/2.0]
        #znew = _interpF(xnew[:,0], ynew[0,:])
        znew = _interpF(xnew, ynew)
        print znew.shape
        print znew
        #plt.pcolor(ix, iy, F)
        #plt.pcolor(ynew, xnew[::-1,:], znew)
        plt.imshow(znew)
        #plt.pcolor(F)
        plt.colorbar()
        plt.show()

    #plt.quiver(delF[0], delF[1])
    #plt.show()


    #import pdb; pdb.set_trace()

    A = 4
    d = scriptN(A)
    # 1.
    u = np.zeros((2, d, d))
    m = 0
    a = 3 
    stepsize = 0.4
    for loop in range(5):
        # 2.
        iP1 = inv_Psi(u[0], a)
        iP2 = inv_Psi(u[1], a)
        def Um(x): 
            return np.array([
                iP1(x),
                iP2(x),
            ])
        #print(Um)

        def W(q, x):
            z = x + Um(x)
            #assert np.fabs(interpDelF[q](z)-interpDelF2[q](z))<0.0001, "Point: {0}, q={3} new = {1}, old = {2}".format(z, interpDelF[q](z), interpDelF2[q](z), q)
            return interpDelF[q](z) * (interpF(z) - fI(x))

        #print(xs)
    
        v = np.zeros((2,)+(scriptN(a),)*2)
        # 4.
        #Wx0 = np.empty((scriptN(a), scriptN(a)) + xs.shape[:2])
        #Wx1 = np.empty((scriptN(a), scriptN(a)) + xs.shape[:2])
        dx = (xs.shape[0]*xs.shape[1])
        for x0 in range(xs.shape[0]):
            for x1 in range(xs.shape[1]):
                x = xs[x0,x1] 
                z = x + Um(x)
                term = (interpF(z) - I[x0,x1])
                #v[0,k1,k2] = 0.0
                #v[1,k1,k2] = 0.0
                for k1 in range(scriptN(a)):
                    for k2 in range(scriptN(a)):
                        #p = psi(k1, k2, x)
                        pterm_div_dx = term * twopi * cos(twopi * (k1 * x[0] + k2 * x[1])) / dx
                        Wx0 = interpDelF[0](z) * pterm_div_dx
                        Wx1 = interpDelF[1](z) * pterm_div_dx
                        v[0,k1,k2] += Wx0
                        v[1,k1,k2] += Wx1

        # Must include psi as well!!

            


        # Calculate cost, just for sanity check
        logprior = 0.0
        for k1 in range(scriptN(a)):
            for k2 in range(scriptN(a)):
                lmbk = (k1**2 + k2**2)
                logprior += lmbk * (u[0,k1,k2]**2 + u[1,k1,k2]**2)
        logprior /= 2.0

        loglikelihood = 0.0
        defs = np.empty(F.shape + (2,))
        for x0 in range(F.shape[0]):
            for x1 in range(F.shape[1]):
                x = xs[x0,x1] 
                z = x + Um(x)
                defs[x0, x1] = Um(x)
                #print z
                # TODO: Um(x) should 
                #loglikelihood += (interpF(z) - fI(x))**2#/(F.shape[0]*F.shape[1])
                loglikelihood += (interpF(z) - fI(x))**2 / (F.shape[0]*F.shape[1])

        print "MAX DEF:", np.max(np.fabs(defs[:,:,0])), np.max(np.fabs(defs[:,:,1]))

        if False and loop == 10:
            plt.quiver(defs[:,:,1], defs[:,:,0])
            plt.show() 

        # Cost function
        J = logprior + loglikelihood
        print "Cost:", J, logprior, loglikelihood

        new_u = np.empty(u.shape)

        # 5. Gradient descent
        for q in range(2):
            for k1 in range(scriptN(a)):
                for k2 in range(scriptN(a)):
                    lmbk = (k1**2 + k2**2)
                    term = (lmbk * u[q,k1,k2] + v[q,k1,k2])
                    new_u[q,k1,k2] = u[q,k1,k2] - stepsize * term
                

        print "========= U (pixels) ==========="
        print u[:,0:scriptN(a),0:scriptN(a)] * F.shape[0] * twopi
        #print new_u
        u = new_u
        #a += 1

        # Print the new deformation as a vector field
        Uvx = np.empty(F.shape)
        Uvy = np.empty(F.shape)
        for x0 in range(F.shape[0]):
            for x1 in range(F.shape[1]):
                x = xs[x0,x1]
                ux = Um(x)
                Uvx[x0,x1] = ux[0]
                Uvy[x0,x1] = ux[1]

        #Q = plt.quiver(Uvx, Uvy)
        #plt.show() 


    return u
     
def deform(I, u):
    """Deform I according to u"""
    im = np.zeros(I.shape)

    xs = np.empty(im.shape + (2,))
    for x0 in range(im.shape[0]):
        for x1 in range(im.shape[1]):
            xs[x0,x1] = np.array([float(x0)/im.shape[0], float(x1)/im.shape[1]])

    xs0 = xs[:,:,0]
    xs1 = xs[:,:,1]

    a = 2 
    iP1 = inv_Psi(u[0], a)
    iP2 = inv_Psi(u[1], a)
    def Um(x): 
        return np.array([
            iP1(x),
            iP2(x),
        ])

    interpI = interp2d(xs0, xs1, I)

    for x0 in range(im.shape[0]): 
        for x1 in range(im.shape[1]): 
            x = xs[x0,x1]
            z = x + Um(x)
            
            im[x0,x1] = interpI(z)
    return im

def main():
    #from amitgroup.io.mnist import read
    #images, _ = read('training', '/local/mnist', [9]) 
    #images = np.load("data/nines.npz")['images'][:,::-1,:]
    #import pylab as plt

    #shifted = np.zeros(images[0].shape)    
    #shifted[:-3,:] = images[0,3:,:]

    #im1, im2 = images[0], shifted
    #im1, im2 = images[1], images[3] 

    im1 = ag.io.load_image('data/Images_0', 45)
    im2 = ag.io.load_image('data/Images_1', 23)

    im1 = im1[::-1,:]
    im2 = im2[::-1,:]

    if 1:
        
        u = np.array([[[-0.48947856, -4.84035154],
                       [-0.03106279, 0.01453099]],
                      [[-0.1421006,  -2.0178281 ],
                       [ 0.18980185, -0.05704723]]])/(twopi*im1.shape[0])
    
        u = imagedef(im1, im2)

        if 0:
            im3 = deform(im1, u)

            d = dict(origin='lower', interpolation='nearest', cmap=plt.cm.gray)
            plt.figure(figsize=(14,6))
            plt.subplot(131)
            plt.title("Prototype")
            plt.imshow(im1, **d)
            plt.subplot(132)
            plt.title("Original")
            plt.imshow(im2, **d) 
            plt.subplot(133)
            plt.title("Deformed")
            plt.imshow(im3, **d)
            plt.show()

    elif 1:
        plt.figure(figsize=(14,6))
        plt.subplot(121)
        plt.title("F")
        plt.imshow(im1, origin='lower')
        plt.subplot(122)
        plt.title("I")
        plt.imshow(im2, origin='lower') 
        plt.show()

    else:
        imagedef(im1, im2)
    

if __name__ == '__main__':
    import cProfile
    cProfile.run('main()')
    #main()
