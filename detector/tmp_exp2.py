
from __future__ import division

import numpy as np
import scipy.special 

def runexp(amount, rescale, seed=0):
    rs = np.random.RandomState(seed)

    #amount = np.asarray([1000001, 100000])
    #densities = np.asarray([1/10, 1/10])
    #means = np.zeros(x0.size)
    #means = np.log(y0)# / pixels0)
    #means = np.zeros(x0.size)

    all_samples = [] 

    for i, N in enumerate(amount):
        samples = np.vstack([rs.normal(size=N), i*np.ones(N)])
        #print samples
        #import pdb; pdb.set_trace()

        for j in xrange(samples.shape[1]):
            samples[0,j] = rescale(samples[0,j], i)
        #samples -= means[i]
        
        all_samples.append(samples)

    all_s = np.hstack(all_samples)

    II = np.argsort(all_s)

    all_s = all_s[:,II[0]]

    top_samples = all_s[:,-1000:]

    counts = np.bincount(top_samples[1].astype(np.int32), minlength=len(amount))
    return counts

EM = 0.57721

data = np.load('ss.npz')

x0, y0 = data['x0'], data['y0']

#print 2**x0[100]
x0 = x0[120:]
y0 = y0[120:]

densities = y0 * (2**x0)**2

pixels0 = 1000000 / (2**x0)**2 
amount = pixels0 * 10
#means = np.log(y0 * /10)
#means = 1.0 * np.log(2**x0)# / np.sqrt(x0)
means = 0.9 * x0
#means = np.zeros(x0.shape)
#np.seterr(all='raise')

print 'smallest size', 2**x0[0]
print 'minimum amount', amount.min()
print 'maximum amount', amount.max()

import scipy.stats as st

Ns = amount / 20.0

mus = np.asarray([st.norm.ppf(1 - 1/Ns[i]) for i in xrange(len(amount))])
sigs = np.asarray([st.norm.ppf(1 - 1/Ns[i]/np.exp(1)) - mus[i] for i in xrange(len(amount))])

def rescale(x, i):
    #mu = st.norm.ppf(1 - 1/amount[i])
    #sig = st.norm.ppf(1 - 1/amount[i]/np.exp(1)) - mu
    mu = mus[i]
    sig = sigs[i]

    mu0 = mu + sig * EM
    sig0 = sig * np.pi / np.sqrt(6)

    #Ireturn x + means[i]
    #return (x - mu) / sig
    #return (x - mu0) / sig0 
    xc = max(x, mu-0.5)
    v = st.genextreme.logcdf(xc, 0, loc=mu, scale=sig)
    if np.isinf(v):
        raise 'Hell'
    return v

def rescale(x, i):
    xi = x0[i]
    #return x + 2 * xi
    return x + np.log(2**(2*xi)) + st.norm.logpdf(xi, loc=7.5, scale=1.6)


x = np.linspace(-30, 30, 300)
start = -30
step = np.diff(x)[0]
points = []

for i in xrange(len(amount)):
    y = []
    for xi in x:
        v = rescale(xi, i)
        y.append(v) 
    y = np.asarray(y)

    points.append(y)
    
#np.savez('st2.npz', start=start, step=step, points=points)

# Store these values
#x = np.linspace(-30, 30, 300)
#for i in xrange(

all_counts = None
for i in xrange(10):
    print 'Running trial', i
    counts = runexp(amount, rescale, seed=i)
    if all_counts is None:
        all_counts = counts
    else:
        all_counts += counts

print all_counts
import pylab as plt
#print counts * densities 
worth = all_counts / densities 
plt.plot(x0, all_counts)
plt.twinx()
plt.plot(x0, densities, color='red')
plt.show()

bins = np.linspace(-10, 10, 200)

if 0:
    plt.hist(all_samples[0][0], bins=bins, alpha=0.5)
    plt.hist(all_samples[1][0], bins=bins, alpha=0.5)
    plt.ylim((0, 1000))
    plt.show()
#samples = 
