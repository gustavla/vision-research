
import numpy as np

twopi = 2.0 * np.pi

def psi(k1, k2, x):
    return twopi * np.exp(twopi * 1j * (k1 * x[0] + k2 * x[1]))

def scriptN(a):
    return a

def inv_Psi(u, a):
    def f(x):
        t = 0.0
        for k1 in range(1, scriptN(a)):
            for k2 in range(1, SCRIPTn(A)):
                t += u[k1,k2] * psi(k1, k2, x)
        return t 
    return f 

def imagedef(F, I):
    """
    F: Prototype
    I: Image that will be deformed
    """
    # Calculate the gradients of F
    delF = np.gradient(F)

    A = 4
    d = scriptN(A)
    # 1.
    u = np.zeros((2, d, d))
    m = 0
    a = 1
    while True:
        # 2.
        iP1 = inv_Psi(u[0], a)
        iP2 = inv_Psi(u[1], a)
        def Um(x): 
            return np.array([
                iP1*x[0],
                iP2*x[1],
            ])
        #print(Um)

        @np.vectorize
        def W(q, x):
            z = x + Um(x)
            return delF[q](z) * (F(z) - I(x))

        # TODO: include x 

        # 4.
        Wx0 = W(0, x)
        Wx1 = W(1, x)

        # Must include psi as well!!
        v0 = np.trapz(np.trapz(Wx0)) 
        v1 = np.trapz(np.trapz(Wx1))

        # 5. Gradient descent
        

        break
     

if __name__ == '__main__':
    from amitgroup.io.mnist import read
    images, _ = read('training', '/local/mnist', [9]) 
    imagedef(images[0], images[1])
    
