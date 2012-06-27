
import numpy as np
from scipy import interpolate
from scipy import linalg
import amitgroup as ag
from copy import copy
from math import cos 

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

def psi(k1, k2, x):
    return twopi * cos(twopi * (k1 * x[0] + k2 * x[1]))

def scriptN(a):
    return a

def U(x, u, a):
    inv_Psi0 = 0.0
    inv_Psi1 = 0.0
    n = scriptN(a)
    for k1 in xrange(n):
        for k2 in xrange(n):
            ps = psi(k1, k2, x)
            inv_Psi0 += u[0,k1,k2] * ps
            inv_Psi1 += u[1,k1,k2] * ps
    return np.array([inv_Psi0, inv_Psi1]) 

def _deform_x(xs, u, a):
    zs = np.empty(xs.shape)
    for x0 in range(xs.shape[0]):
        for x1 in range(xs.shape[1]):
            x = xs[x0,x1] 
            z = x + U(x, u, a)
            zs[x0,x1] = z
    return zs

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
    # TODO: Does not handle rectangular images
    delF[0] /= F.shape[0]
    delF[1] /= F.shape[0]
     
    # 1.
    rho = 1.0 
    A = 3 
    d = scriptN(A)
    u = np.zeros((2, d, d))
    u[:,0,0] = 3.0/twopi/32.0
    u[0,1,0] = 1.0/twopi/32.0
    m = 0
    a = 0 
    stepsize = 0.3
    dx = 1.0/(xs.shape[0]*xs.shape[1])
    for a in range(1, A+1):
        n = scriptN(a)
        for loop_inner in range({1: 150, 2:500, 3:0}[a]):
            # 2.

            # Calculate deformed xs
            zs = _deform_x(xs, u, a)

            # Interpolated F at zs
            Fzs = ag.math.interp2d(zs, F, startx=np.zeros(2))

            # Interpolate delF at zs 
            delFzs = np.empty((2,) + F.shape) 
            for q in range(2):
                delFzs[q] = ag.math.interp2d(zs, delF[q])

            v = np.zeros((2,)+(n,)*2)
            # 4.
            terms = Fzs - I
            for x0 in range(xs.shape[0]):
                for x1 in range(xs.shape[1]):
                    x = xs[x0,x1] 
                    term = terms[x0,x1]#Fzs[x0,x1] - I[x0,x1]
                    for k1 in range(n):
                        for k2 in range(n):
                            #p = psi(k1, k2, x)
                            #pterm = term * cos(k1 * x2pi[0] + k2 * x2pi[1])
                            pterm = term * psi(k1, k2, x)
                            v[:,k1,k2] += delFzs[:,x0,x1] * pterm 

            # We didn't multiply by this
            v *= dx# * twopi

            # Must include psi as well!!

                


            # Calculate cost, just for sanity check
            if 1:
                logprior = 0.0
                for k1 in range(n):
                    for k2 in range(n):
                        lmbk = (k1**2 + k2**2)**rho
                        logprior += lmbk * (u[0,k1,k2]**2 + u[1,k1,k2]**2)
                logprior /= 2.0

                loglikelihood = 0.0
                for x0 in range(F.shape[0]):
                    for x1 in range(F.shape[1]):
                        loglikelihood += (Fzs[x0,x1] - I[x0,x1])**2 
                loglikelihood *= dx

                if False and loop_outer == 10:
                    plt.quiver(defs[:,:,1], defs[:,:,0])
                    plt.show() 

                # Cost function
                J = logprior + loglikelihood
                print "Cost:", J, logprior, loglikelihood

            new_u = np.copy(u)

            # 5. Gradient descent
            for q in range(2):
                for k1 in range(n):
                    for k2 in range(n):
                        lmbk = float(k1**2 + k2**2)**rho
                        term = (lmbk * u[q,k1,k2] + v[q,k1,k2])
                        new_u[q,k1,k2] = u[q,k1,k2] - stepsize * term
                    

            print "========= U (pixels) ==========="
            print u[:,0:n,0:n] * F.shape[0] * twopi
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
    zs = _deform_x(xs, u, a)
    im = ag.math.interp2d(zs, I)
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
    u = np.zeros((2,3,3))
    u[:,0,0] = 3.0/twopi/32.0
    u[0,1,0] = 1.0/twopi/32.0
    #u = np.array([3.0, 3.0])/twopi/im1.shape[0]

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
