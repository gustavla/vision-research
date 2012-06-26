import numpy as np
import matplotlib.pylab as plt
import amitgroup as ag

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

N = 5 
x, y = np.mgrid[0:N,0:N]
z = np.zeros((N,N))
z[0,0] = 1.0
z[1,0] = 0.3 
f1 = interp2d(x, y, z, fill_value=0.0)
f2 = ag.math.interp2d(z, fill_value=None)

xnew, ynew = np.mgrid[-1:N:100j, -1:N:100j]
znew1 = np.empty((100, 100))
znew2 = np.empty((100, 100))

for i in range(100):
    for j in range(100):
        znew1[i,j] = f1(np.array([xnew[i,0],ynew[0,j]])) 
        znew2[i,j] = f2(np.array([xnew[i,0],ynew[0,j]])) 

plt.figure(figsize=(14,6))
plt.subplot(131)
plt.pcolor(x, y, z)
plt.colorbar()
plt.subplot(132)
plt.pcolor(xnew, ynew, znew1)
plt.colorbar()
plt.subplot(133)
plt.pcolor(xnew, ynew, znew2)
plt.colorbar()
plt.show()
