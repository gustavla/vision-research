
import numpy as np
from scipy import interpolate
from scipy import linalg
import amitgroup as ag
from copy import copy
from math import cos 
from itertools import product
import pywt
PLOT = True 
#PLOT = False
if PLOT: 
    import matplotlib.pylab as plt

TEST = False
TEST = True

def main():
    #import amitgroup.io.mnist
    #images, _ = read('training', '/local/mnist', [9]) 
    #shifted = np.zeros(images[0].shape)    
    #shifted[:-3,:] = images[0,3:,:]

    im1, im2 = np.zeros((32, 32)), np.zeros((32, 32))

    wl_name = "db2"
    levels = len(pywt.wavedec(range(32), wl_name)) - 1
    scriptNs = map(len, pywt.wavedec(range(32), wl_name, level=levels))

    if 0:
        images = np.load("data/nines.npz")['images']
        im1[:28,:28] = images[0]
        im2[:28,:28] = images[2]
    else:
        im1 = ag.io.load_image('data/Images_0', 45)
        im2 = ag.io.load_image('data/Images_1', 23)

    im1 = im1[::-1,:]
    im2 = im2[::-1,:]

    A = 2 
    if 0:
        u = []
        for q in range(2):
            u0 = [np.zeros((1,1))]
            for s in range(0, A):
                u0.append((np.zeros((2**s,2**s)), np.zeros((2**s,2**s)), np.zeros((2**s,2**s))))
            u.append(u0)

        #u[0][1][0][0] += 0.5 
        #u[1][1][1][0] += 0.2 
        
        #u[0][1:][2][0][0,3] = 0.2 

        #print u[0][1:][2][0]

        xs = np.empty((20, 20, 2))
        for x0 in range(xs.shape[0]):
            for x1 in range(xs.shape[1]):
                xs[x0,x1] = np.array([x0/float(xs.shape[0]), x1/float(xs.shape[1])])
        
        defx = ag.ml.deform_map(xs, u, scriptNs)

        imdef = deform(im1, u, scriptNs)
        d = dict(origin='lower', interpolation='nearest', cmap=plt.cm.gray)

        plt.figure(figsize=(14,4))
        plt.subplot(131)
        plt.title("Original")
        plt.imshow(im1, **d) 
        plt.subplot(132)
        plt.title("Deformed")
        plt.imshow(imdef, **d)
        plt.subplot(133)
        plt.title("Deform map")
        plt.quiver(xs[:,:,1], xs[:,:,0], defx[:,:,1], defx[:,:,0])
        plt.show()
        
    elif 1:
        # Blur images
        im1b = im1#ag.math.blur_image(im1, 2)
        im2b = im2#ag.math.blur_image(im2, 2)

        show_costs = True

        u, info = ag.ml.imagedef(im1b, im2b, rho=3.0, calc_costs=show_costs)


        if PLOT and show_costs:
            logpriors = info['logpriors']
            loglikelihoods = info['loglikelihoods']
            np.savez('logs', logpriors=logpriors, loglikelihoods=loglikelihoods)

            plotfunc = plt.semilogy
            plt.figure(figsize=(8,12))
            plt.subplot(211)
            costs = logpriors + loglikelihoods
            plotfunc(costs, label="J")
            plotfunc(loglikelihoods, label="log likelihood")
            plt.legend()
            plt.subplot(212) 
            plotfunc(logpriors, label="log prior")
            plt.legend()
            plt.show()

    
        if u is not None:
            im3 = ag.ml.deform(im1, u, scriptNs)

        if PLOT:
            d = dict(origin='lower', interpolation='nearest', cmap=plt.cm.gray)

            plt.figure(figsize=(16,6))
            plt.subplot(141)
            plt.title("Prototype")
            plt.imshow(im1, **d)
            plt.subplot(142)
            plt.title("Original")
            plt.imshow(im2, **d) 
            plt.subplot(143)
            plt.title("Deformed")
            plt.imshow(im3, **d)
            plt.subplot(144)
            plt.title("Deformed")
            plt.imshow(im2-im3, **d)
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


    import pickle
    if TEST:
        im3correct = pickle.load(open('im3.p', 'rb'))
        if u is not None:
            passed = (im3 == im3correct).all()
            print "PASSED:", ['NO', 'YES'][passed] 
            print ((im3 - im3correct)**2).sum()
    else:
        pickle.dump(im3, open('im3.p', 'wb'))
    

if __name__ == '__main__':
    import cProfile
    #cProfile.run('main()')
    main()
