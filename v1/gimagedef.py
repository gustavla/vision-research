
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

def deformfield(xs, ys, u, a):
    iP1 = inv_Psi(u[0], a)
    iP2 = inv_Psi(u[1], a)
    def Um(x): 
        return np.array([
            iP1(x),
            iP2(x),
        ])

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

    delF = np.gradient(F)
    delF[0] /= F.shape[0]
    delF[1] /= F.shape[0]
    ix, iy = np.mgrid[0:F.shape[0]-1:1j*F.shape[0], 0:F.shape[1]-1:1j*F.shape[1]]
    

     
    #interpF = ag.math.interp2d_func(F)
    #interpF.dx = 1.0/np.array(F.shape)

    interpDelF = []
    for q in range(len(delF)):
        interpDelF_f = ag.math.interp2d_func(delF[q])
        interpDelF_f.dx = 1.0/np.array(F.shape)
        interpDelF.append(interpDelF_f)

    A = 3 
    d = scriptN(A)
    # 1.
    u = np.zeros((2, d, d))
    m = 0
    a = 1 
    stepsize = 0.2
    dx = 1.0/(xs.shape[0]*xs.shape[1])
    for loop in range(200):
        # 2.
        iP1 = inv_Psi(u[0], a)
        iP2 = inv_Psi(u[1], a)

        # Calculate deformed xs
        zs = np.empty(xs.shape)
        for x0 in range(xs.shape[0]):
            for x1 in range(xs.shape[1]):
                x = xs[x0,x1] 
                z = x + np.array([iP1(x), iP2(x)]) 
                zs[x0,x1] = z

        # Interpolated F at zs
        import sys
        Fzs = ag.math.interp2d(zs, F, startx=np.zeros(2))

        v = np.zeros((2,)+(scriptN(a),)*2)
        # 4.
        for x0 in range(xs.shape[0]):
            for x1 in range(xs.shape[1]):
                x = xs[x0,x1] 
                x2pi = twopi * x
                z = zs[x0,x1]
                #assert np.fabs(interpF(z)-Fzs[x0,x1])<0.0000001, "z = {0} interpF = {1} Fzs = {2}, {3}".format(z, interpF(z), Fzs[x0,x1], interpF.dx)
                term = (Fzs[x0,x1] - I[x0,x1])
                for k1 in range(scriptN(a)):
                    for k2 in range(scriptN(a)):
                        #p = psi(k1, k2, x)
                        pterm = term * cos(k1 * x2pi[0] + k2 * x2pi[1])
                        Wx0 = interpDelF[0](z) * pterm
                        Wx1 = interpDelF[1](z) * pterm
                        v[0,k1,k2] += Wx0
                        v[1,k1,k2] += Wx1

        # We didn't multiply by this
        v *= dx * twopi

        # Must include psi as well!!

            


        # Calculate cost, just for sanity check
        if 0:
            logprior = 0.0
            for k1 in range(scriptN(a)):
                for k2 in range(scriptN(a)):
                    lmbk = (k1**2 + k2**2)
                    logprior += lmbk * (u[0,k1,k2]**2 + u[1,k1,k2]**2)
            logprior /= 2.0

            loglikelihood = 0.0
            for x0 in range(F.shape[0]):
                for x1 in range(F.shape[1]):
                    x = xs[x0,x1] 
                    z = zs[x0,x1]
                    loglikelihood += (interpF(z) - I[x0,x1])**2 / (F.shape[0]*F.shape[1])

            if False and loop == 10:
                plt.quiver(defs[:,:,1], defs[:,:,0])
                plt.show() 

            # Cost function
            J = logprior + loglikelihood
            print "Cost:", J, logprior, loglikelihood

        new_u = np.copy(u)

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
        if 0:
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

    a = u.shape[0]
    #a = u.size-2
    iP1 = inv_Psi(u[0], a)
    iP2 = inv_Psi(u[1], a)
    def Um(x): 
        return np.array([
            iP1(x),
            iP2(x),
        ])

    interpI = ag.math.interp2d_func(I)
    interpI.dx = 1.0/np.array(I.shape)

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
    PLOT = False 
    if PLOT: 
        import pylab as plt

    #shifted = np.zeros(images[0].shape)    
    #shifted[:-3,:] = images[0,3:,:]

    #im1, im2 = images[0], shifted
    #im1, im2 = images[1], images[3] 

    im1 = ag.io.load_image('data/Images_0', 45)
    im2 = ag.io.load_image('data/Images_1', 23)

    im1 = im1[::-1,:]
    im2 = im2[::-1,:]

    u = np.array(2*range(9)[::-1]).reshape((2,3,3))/2000.0

    im2 = deform(im1, u)    

    if 1:
        if 0:
            u = np.array([[[-0.48947856, -4.84035154],
                           [-0.03106279, 0.01453099]],
                          [[-0.1421006,  -2.0178281 ],
                           [ 0.18980185, -0.05704723]]])/(twopi*im1.shape[0])
    
        u = imagedef(im1, im2)

        if PLOT:
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

        im3 = deform(im2, u)

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
