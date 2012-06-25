

import numpy as np
import matplotlib.pylab as plt
from scipy import interpolate

def lerp(a, x, y):
    return y * a + (1-a) * x 

def interp2d(x, y, z, fill_value=0.0):
    @np.vectorize
    def f(x0, y0):
        for i in range(len(x[:,0])-1):
            for j in range(len(y[0,:])-1):
                if x[i+1,0] >= x0 >= x[i,0] and y[0,j+1] >= y0 >= y[0,j]:
                    xp1 = lerp((x0-x[i,0])/(x[i+1,0]-x[i,0]), z[i,j], z[i+1,j])
                    xp2 = lerp((x0-x[i,0])/(x[i+1,0]-x[i,0]), z[i,j+1], z[i+1,j+1])

                    print y0, y[0,j], y[0,j+1]
                    return lerp((y0-y[0,j])/(y[0,j+1]-y[0,j]), xp1, xp2)

        return fill_value 
    return f


if 1:
    x,y = np.mgrid[-1:1:20j,-1:1:20j]
    z = (x+y)*np.exp(-6.0*(x*x+y*y))

    plt.figure(figsize=(14,6))
    plt.subplot(121)
    plt.pcolor(x,y,z)
    plt.colorbar()
    plt.title("Sparsely sampled function.")

    xnew,ynew = np.mgrid[-1:1:170j,-1:1:170j]
    #tck = interpolate.bisplrep(x,y,z,s=0)
    #znew = interpolate.bisplev(xnew[:,0],ynew[0,:],tck)
    f = interp2d(x,y,z)
    znew = f(xnew, ynew)
else:
    x, y = np.mgrid[0:1:4j,0:1:4j]
    z = np.zeros(x.shape)
    z[1,1] = 1.0

    print x
    print z

    plt.figure(figsize=(14,6))
    plt.subplot(121)
    plt.pcolor(x,y,z)
    #plt.pcolor(z)
    plt.colorbar()
    plt.title("Sparsely sampled function.")

    xnew,ynew = np.mgrid[0:1:100j,0:1:100j]
    #tck = interpolate.bisplrep(x,y,z,s=0)
    #znew = interpolate.bisplev(xnew[:,0],ynew[0,:],tck)
    f = interp2d(x,y,z)
    znew = f(xnew, ynew)

print znew.shape

plt.subplot(122)
plt.pcolor(xnew,ynew,znew)
#plt.pcolor(znew)
plt.colorbar()
plt.title("Interpolated function.")


plt.show()
