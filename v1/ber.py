from __future__ import print_function
from __future__ import division

import amitgroup as ag
import amitgroup.io
import amitgroup.features
import amitgroup.ml
import sys
import numpy as np
import matplotlib.pylab as plt

PLOT = '--plot' in sys.argv

def add_blur(x):
    y = np.copy(x)
    x[:,1:,:] += y[:,:-1,:]
    x[:,:-1,:] += y[:,1:,:]
    x[:,:,1:] += y[:,:,:-1]
    x[:,:,:-1] += y[:,:,1:]

    x[:,1:,1:] += y[:,:-1,:-1]
    x[:,:-1,:-1] += y[:,1:,1:]
    x[:,1:,:-1] += y[:,:-1,1:]
    x[:,:-1,1:] += y[:,1:,:-1]
        
    #return x 

def features():
    digits_data = []
    for digit in range(10):
        print("digit {0}".format(digit))
        data, _ = ag.io.load_mnist("training", [digit])
        # Extract features
        edges = ag.features.bedges(data)
        digit_F = np.rollaxis(edges.sum(axis=0), 2)
        #add_blur(digit_F)
        digit_F += 1
        # Normalize
        digit_F = digit_F.astype(np.float64)/data.shape[0]
        digits_data.append(digit_F)
    np.savez("F.npz", F=np.array(digits_data)) 

def main(F, coef, filename=None, test_index=1):
        
    #Fmax = F.max()
    #F *= 1.0
    #F = np.clip(F, 0.01, 1.0 - 0.01)
    #print("Normalizing with {0}".format(Fmax))
    #testing = ag.io.load_example('mnist')
    testing = np.load('digits.npz')['nines']
    testing = testing[:,::-1]

    settings = dict(origin='lower', interpolation='nearest', cmap=None, vmin=0.0, vmax=1.0)
    feat = int(sys.argv[1]) 

    costs = []
    for d in [9]: 
        print("Digit:", d)
        imdef, info = ag.ml.bernoulli_model(F[d], testing[test_index], calc_costs=True, tol=1e-4)
        #imdef, info = ag.ml.imagedef(F[d], testing[test_index], coef=1e-2, stepsize=0.3, calc_costs=True)


        print(info['iterations_per_level'])
        print("Cost:", -info['loglikelihoods'][-1] - info['logpriors'][-1])

        subject = ag.features.bedges(testing[test_index])[:,:,feat].astype(np.float64)

        #subject = testing[1]
        #imdef = ag.util.DisplacementFieldWavelet(testing[0].shape)
        #imdef.u[0][0,0] = 0.6 
        x, y = imdef.get_x(subject.shape)
        Ux, Uy = imdef.deform_map(x, y)

        print('U-max:',Ux.max(), Uy.max())
        #print(imdef.u)

        imf = imdef.deform(subject) 
        canon_nine = np.load('canonical-digits.npz')['F'][9]
        #im = imdef.deform(F[d,feat])
        im = imdef.deform(canon_nine)
       
        costs.append(-info['loglikelihoods'][-1] - info['loglikelihoods'][-1])

        if PLOT:
            plt.figure(figsize=(11,7))
            plt.subplot(231)
            plt.imshow(subject, **settings)
            plt.subplot(232)
            plt.imshow(imf, **settings)
            plt.subplot(233)
            #plt.imshow(F[d,feat], **settings)
            plt.imshow(F[d,feat], **settings)
            plt.subplot(234)
            plt.quiver(y, x, Uy, Ux, scale=1.0) 
            plt.subplot(235)
            plt.imshow(testing[test_index], **settings)
            plt.subplot(236)
            plt.imshow(im, **settings)
            plt.colorbar()
            #plt.plot(-(info['loglikelihoods']+info['logpriors']))

    print("RESULT Digit: {0}".format(np.argmin(costs)))

    if PLOT:
        if filename:
            plt.savefig(filename)
        else:
            plt.show()

if __name__ == '__main__':
    try:
        option = sys.argv[1]
    except IndexError:
        option = None
        
    if option == 'features':
        features()
    else:
        F = np.load("F.npz")['F']
        F = F[:,:,::-1]
        print(F.shape)
        #F = np.load("canonical-digits.npz")['F']
        #F = F[:,::-1]
        if 0:
            for i in range(5):
                coef = 0.001
                main(F, coef, 'nine{0}.png'.format(i), test_index=i)    
        elif 1:
            coef = 0.05 
            main(F, coef, test_index=0) 
        else:
            for i, coef in enumerate(np.linspace(0.0001, 0.01, 50)):
                print("Doing coef {0}".format(coef))
                main(F, coef, 'pic-{0:03}.png'.format(i))
