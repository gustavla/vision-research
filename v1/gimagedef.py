
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
        
    if 1:
        # Blur images
        im1b = im1#ag.math.blur_image(im1, 2)
        im2b = im2#ag.math.blur_image(im2, 2)

        show_costs = True

        imgdef, info = ag.ml.imagedef(im1b, im2b, rho=3.0, calc_costs=show_costs)


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

    
        im3 = imgdef.deform(im1)

        if PLOT:
            d = dict(origin='lower', interpolation='nearest', cmap=plt.cm.gray)

            plt.figure(figsize=(9,9))
            plt.subplot(221)
            plt.title("Prototype")
            plt.imshow(im1, **d)
            plt.subplot(222)
            plt.title("Original")
            plt.imshow(im2, **d) 
            plt.subplot(223)
            plt.title("Deformed")
            plt.imshow(im3, **d)
            
            plt.subplot(224)
            if 0:
                plt.title("Deformed")
                plt.imshow(im2-im3, **d)
                plt.colorbar()
            else:
                plt.title("Deformation map")
                x, y = imgdef.get_x(im1.shape)
                Ux, Uy = imgdef.deform_map(x, y) 
                plt.quiver(y, x, Uy, Ux)
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
        passed = (im3 == im3correct).all()
        print "PASSED:", ['NO', 'YES'][passed] 
        print ((im3 - im3correct)**2).sum()
    else:
        pickle.dump(im3, open('im3.p', 'wb'))
    

if __name__ == '__main__':
    import cProfile
    #cProfile.run('main()')
    main()
