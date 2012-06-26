
import numpy as np
from scipy import interpolate
from scipy import linalg

twopi = 2.0 * np.pi

def lerp(a, x, y):
    return y * a + (1-a) * x 

def interp2d(x, y, z, fill_value=0.0):
 #   @np.vectorize
    def f(p):
        for i in range(len(x[:,0])-1):
            for j in range(len(y[0,:])-1):
                if x[i+1,0] >= p[0] >= x[i,0] and y[0,j+1] >= p[1] >= y[0,j]:
                    xp1 = lerp((p[0]-x[i,0])/(x[i+1,0]-x[i,0]), z[i,j], z[i+1,j])
                    xp2 = lerp((p[0]-x[i,0])/(x[i+1,0]-x[i,0]), z[i,j+1], z[i+1,j+1])
                    return lerp((p[1]-y[0,j])/(y[0,j+1]-y[0,j]), xp1, xp2)
        return fill_value 
    return f

def deformfield(xs, ys, u, a):
    iP1 = inv_Psi(u[0], a)
    iP2 = inv_Psi(u[1], a)
    def Um(x): 
        return np.array([
            iP1(x),
            iP2(x),
        ])

def psi(k1, k2, x):
    return twopi * np.real(np.exp(twopi * 1j * (k1 * x[0] + k2 * x[1])))

def scriptN(a):
    return a

def inv_Psi(uxi, a):
    def f(x):
        t = 0.0
        for k1 in range(scriptN(a)):
            for k2 in range(scriptN(a)):
                #print k1, k2
                t += uxi[k1,k2] * psi(k1, k2, x)
        return t 
    return f 

def imagedef(F, I):
    """
    F: Prototype
    I: Image that will be deformed
    """
    xs = np.empty(F.shape + (2,))
    for x0 in range(F.shape[0]):
        for x1 in range(F.shape[1]):
            xs[x0,x1] = np.array([float(x0)/F.shape[0], float(x1)/F.shape[1]])

    xs0 = xs[:,:,0]
    xs1 = xs[:,:,1]

    px = xs1
    py = xs0

    # Calculate the gradients of F 
    # (the gradient does not know dx from just F, so we must multiply by dx)
    delF = np.gradient(F)
    delF[0] /= F.shape[0]
    delF[1] /= F.shape[1]
    ix, iy = np.mgrid[0:F.shape[0]-1:1j*F.shape[0], 0:F.shape[1]-1:1j*F.shape[1]]
    interpF = interp2d(xs[:,:,0], xs[:,:,1], F)
    #_interpF = interp2d(ix, iy, F)
    #tck = interpolate.bisplrep(ix, iy, F, s=0)
    #def interpF(x):
        #return _interpF(x[0]*F.shape[0], x[1]*F.shape[1])[0]
        #return _interpF(x[0]*F.shape[0], x[1]*F.shape[1])[0]
    #    return 0.0

    def fI(x):
        x, y = int(round(x[0]*F.shape[0])), int(round(x[1]*F.shape[1])) 
        return I[x,y]

    interpDelF = []
    for q in range(len(delF)):
        f = interp2d(xs[:,:,0], xs[:,:,1], delF[q])
        interpDelF.append(f)
    
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
    a = 2
    stepsize = 0.4
    for loop in range(100):
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
            return interpDelF[q](z) * (interpF(z) - fI(x))

        #print(xs)
    
        v = np.empty((2,)+(scriptN(a),)*2)
        # 4.
        #Wx0 = np.empty((scriptN(a), scriptN(a)) + xs.shape[:2])
        #Wx1 = np.empty((scriptN(a), scriptN(a)) + xs.shape[:2])
        dx = (xs.shape[0]*xs.shape[1])
        for k1 in range(scriptN(a)):
            for k2 in range(scriptN(a)):
                v[0,k1,k2] = 0.0
                v[1,k1,k2] = 0.0
                for x0 in range(xs.shape[0]):
                    for x1 in range(xs.shape[1]):
                        x = xs[x0,x1] 
                        p = psi(k1, k2, x)
                        Wx0 = p * W(0, x)
                        Wx1 = p * W(1, x)
                        v[0,k1,k2] += Wx0 / dx 
                        v[1,k1,k2] += Wx1 / dx

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

        if loop == 10:
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

if __name__ == '__main__':
    #from amitgroup.io.mnist import read
    #images, _ = read('training', '/local/mnist', [9]) 
    images = np.load("nines.npz")['images'][:,::-1,:]
    import pylab as plt

    shifted = np.zeros(images[0].shape)    
    shifted[:-3,:] = images[0,3:,:]

    im1, im2 = images[0], shifted
    im1, im2 = images[1], images[3] 

    if 1:
        u = np.array([[[-0.48947856, -4.84035154],
                       [-0.03106279, 0.01453099]],
                      [[-0.1421006,  -2.0178281 ],
                       [ 0.18980185, -0.05704723]]])/(twopi*images[0].shape[0])

        im3 = deform(im2, u)

        d = dict(origin='lower', interpolation='nearest')
        plt.figure(figsize=(14,6))
        plt.subplot(221)
        plt.title("Prototype")
        plt.imshow(im1, **d)
        plt.subplot(222)    
        plt.quiver(
        plt.subplot(223)
        plt.title("Original")
        plt.imshow(im2, **d) 
        plt.subplot(224)
        plt.title("Deformed")
        plt.imshow(im3, **d)
        plt.colorbar()
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
    
