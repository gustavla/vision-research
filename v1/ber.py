from __future__ import print_function
from __future__ import division

import amitgroup as ag
import amitgroup.io
import amitgroup.features
import amitgroup.ml
import sys
import numpy as np
import matplotlib.pylab as plt

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

def main():
    F = np.load("F.npz")['F']
    F = F[:,:,::-1]
    testing = ag.io.load_example('mnist')
    testing = testing[:,::-1]

    settings = dict(origin='lower', interpolation='nearest', cmap=plt.cm.gray)
    for d in [9]: 
        print("Digit:", d)
        imdef, info = ag.ml.bernoulli_model(testing[0], F[d], calc_costs=True)
        print(info['iterations_per_level'])
        print(-info['loglikelihoods'][-1] - info['logpriors'][-1])

        subject = testing[1]
        #imdef = ag.util.DisplacementFieldWavelet(testing[0].shape)
        #imdef.u[0][0,0] = 0.6 
        x, y = imdef.get_x(subject.shape)
        Ux, Uy = imdef.deform_map(x, y)

        print('U-max:',Ux.max(), Uy.max())
        #print(imdef.u)

        im = imdef.deform(subject) 

        plt.figure(figsize=(7,7))
        plt.subplot(221)
        plt.imshow(subject, **settings)
        plt.subplot(222)
        plt.imshow(im, **settings)
        plt.subplot(223)
        plt.imshow(F[d].sum(axis=0), **settings)
        plt.subplot(224)
        plt.quiver(y, x, Uy, Ux) 
        #plt.plot(-(info['loglikelihoods']+info['logpriors']))

    plt.show()

if __name__ == '__main__':
    try:
        option = sys.argv[1]
    except IndexError:
        option = None
        
    if option == 'features':
        features()
    else:
        main()
