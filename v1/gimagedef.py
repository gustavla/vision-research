
import numpy as np
from scipy import interpolate
from scipy import linalg

twopi = 2.0 * np.pi

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
    # Calculate the gradients of F
    delF = np.gradient(F)
    ix, iy = np.mgrid[0:F.shape[0]-1:1j*F.shape[0], 0:F.shape[1]-1:1j*F.shape[1]]
    interpF = interpolate.interp2d(ix, iy, F, kind='cubic')
    interpDelF = []
    for q in range(len(delF)):
        interpDelF.append(interpolate.interp2d(ix, iy, delF[q], kind='cubic'))

    #import pdb; pdb.set_trace()

    xs = np.empty(F.shape + (2,))
    for x0 in range(F.shape[0]):
        for x1 in range(F.shape[1]):
            xs[x0,x1] = np.array([float(x0)/F.shape[0], float(x1)/F.shape[1]])

    A = 4
    d = scriptN(A)
    # 1.
    u = np.zeros((2, d, d))
    m = 0
    a = 2
    stepsize = 0.1
    for loop in range(8):
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
            #print 'hj',z
            #print(q, x)
            return interpDelF[q](*z)[0] * (interpF(*z)[0] - I[x[0],x[1]])

        #print(xs)
    
        # 4.
        Wx0 = np.empty((scriptN(a), scriptN(a)) + xs.shape[:2])
        Wx1 = np.empty((scriptN(a), scriptN(a)) + xs.shape[:2])
        for k1 in range(scriptN(a)):
            for k2 in range(scriptN(a)):
                for i in range(xs.shape[0]):
                    for j in range(xs.shape[1]):
                        p = psi(k1, k2, (i, j))
                        Wx0[k1,k2,i,j] = p * W(0, xs[i,j])
                        Wx1[k1,k2,i,j] = p * W(1, xs[i,j])

        # Must include psi as well!!


        v = np.empty((2,)+(scriptN(a),)*2)
        
        for k1 in range(scriptN(a)):
            for k2 in range(scriptN(a)):
                #v[0,k1,k2] = np.trapz(np.trapz(Wx0[k1,k2])) 
                #v[1,k1,k2] = np.trapz(np.trapz(Wx1[k1,k2]))
                v[0,k1,k2] = 0.0
                v[1,k1,k2] = 0.0
                for x0 in range(xs.shape[0]):
                    for x1 in range(xs.shape[1]):
                        x = np.array([float(x0)/F.shape[0], float(x1)/F.shape[1]])
                        v[0,k1,k2] += Wx0[k1,k2,x0,x1] * xs.shape[0]*xs.shape[1] 
                        v[1,k1,k2] += Wx1[k1,k2,x0,x1] * xs.shape[0]*xs.shape[1] 
            


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
                x = np.array([float(x0)/F.shape[0], float(x1)/F.shape[1]])
                z = x + Um(x)
                defs[x0, x1] = Um(x)
                #print z
                loglikelihood += (interpF(z[0],z[1])[0] - I[x0,x1])**2#/(F.shape[0]*F.shape[1])

        #print "MAX DEF:", np.max(defs[:,:,0]), np.max(defs[:,:,1])

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
                

        print "========= U ==========="
        print u[:,0:scriptN(a),0:scriptN(a)]
        #print new_u
        u = new_u
        #a += 1

        # Print the new deformation as a vector field
        Uvx = np.empty(F.shape)
        Uvy = np.empty(F.shape)
        for x0 in range(F.shape[0]):
            for x1 in range(F.shape[1]):
                x = np.array([float(x0)/F.shape[0], float(x1)/F.shape[1]])
                ux = Um(x)
                Uvx[x0,x1] = ux[0]
                Uvy[x0,x1] = ux[1]

        Q = plt.quiver(Uvx, Uvy)
        plt.show() 

     

if __name__ == '__main__':
    #from amitgroup.io.mnist import read
    #images, _ = read('training', '/local/mnist', [9]) 
    images = np.load("/local/nines.npz")['images']
    import pylab as plt
    if 0:
        plt.figure()
        plt.subplot(211)
        plt.imshow(images[0])
        plt.subplot(212)
        plt.imshow(images[1]) 
        plt.show()
    imagedef(images[0], images[1])
    
