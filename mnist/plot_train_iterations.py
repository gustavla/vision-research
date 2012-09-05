from __future__ import division
import sys
import argparse

parser = argparse.ArgumentParser(description='MNIST classifier')
parser.add_argument('coefficients', metavar='<coef file>', type=argparse.FileType('rb'), help='Filename of model coefficients')
parser.add_argument('-i', '--index', nargs=3, metavar=('DIGIT', 'MIXTURE', 'AXIS'), default=(0, 0, 0), type=int, help='Index of data, with choice of DIGIT, MIXTURE and AXIS. Defaults to (0, 0, 0).')
parser.add_argument('-n', dest='iterations', nargs=1, metavar='ITERATIONS', default=[sys.maxint], type=int, help='Number of iterations to plot')

args = parser.parse_args()
coef_file = args.coefficients
N = args.iterations[0]
digit, mixture, axis = args.index

import amitgroup as ag
import numpy as np
import matplotlib.pylab as plt

data = np.load(coef_file)

try:
    variances = data['all_iterations_var']
    means = data['all_iterations_mean']
except KeyError:
    raise ValueError("Coefficient file must be run trained with several iterations for this plot")

iterations = min(len(means), N)

def color(i):
    #return (1, 0.7*(1-1*i/iterations), 0)
    f = i/(iterations-1)
    return (f, 0, 1-f)

fig = plt.figure(figsize=(7, 9))
plt.subplot(211)
num_coefs = 0
for i in xrange(iterations):
    flattened = ag.util.wavelet.smart_flatten(means[i,digit,mixture,axis])
    if i == 0 or i == iterations-1:
        label = "Iteration {0}".format(i)
    else:
        label = None
    plt.plot(flattened, label=label, c=color(i))
    num_coefs = len(flattened)
plt.xlim((0, num_coefs-1))
#plt.xlabel('Coefficient')
plt.ylabel('Mean $\mu$')
plt.legend(loc=0)

plt.subplot(212)
num_coefs = 0
for i in xrange(iterations):
    flattened = 1/ag.util.wavelet.smart_flatten(variances[i,digit,mixture,axis])
    plt.semilogy(flattened, label="Iteration {0}".format(i), c=color(i))
    num_coefs = len(flattened)
plt.xlim((0, num_coefs-1))
plt.xlabel('Coefficient')
plt.ylabel('Precision $\lambda$')

plt.show()
