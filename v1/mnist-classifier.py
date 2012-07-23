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

def classify(I, F, filename=None):
    settings = dict(origin='lower', interpolation='nearest', cmap=None, vmin=0.0, vmax=1.0)

    costs = []
    for d in range(10):
        #imdef, info = ag.ml.bernoulli_model(testing[test_index], F[d], jcoef=coef, rho=2.5, stepsize=0.01*1e-3/coef, calc_costs=True, tol=1e-3)
        imdef, info = ag.ml.imagedef(F[d], I, coef=1e-2, stepsize=0.7, calc_costs=True)

        #imdef = ag.util.DisplacementFieldWavelet(testing[0].shape)
        #imdef.u[0][0,0] = 0.6 
        x, y = imdef.get_x(I.shape)
        Ux, Uy = imdef.deform_map(x, y)

        im = imdef.deform(F[d])
       
        costs.append(-info['loglikelihoods'][-1] - info['loglikelihoods'][-1])

    return np.argmin(costs)

if __name__ == '__main__':
    try:
        option = sys.argv[1]
    except IndexError:
        option = None
        
    if option == 'features':
        features()
    else:
        F = np.load("canonical-digits.npz")['F']
        digits, labels = ag.io.load_mnist('testing')
        
        N = len(digits)
        corrects = 0
        for i in range(N): 
            label = classify(digits[i], F)
            correct = int(label == labels[i])
            corrects += correct 
            print("{0} of {1}: {2}".format(i, N, ['fail', 'correct'][correct]))

        print("Precision: {0}".format(correct/N))
