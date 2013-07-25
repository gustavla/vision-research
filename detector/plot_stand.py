from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='Plot llhs')
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')
parser.add_argument('--adjusted', action='store_true')

args = parser.parse_args()
model_file = args.model
adjusted = args.adjusted

import numpy as np
import gv
import matplotlib.pylab as plt

detector = gv.Detector.load(model_file)

L = len(detector.standardization_info)

neg_mn = np.min([np.min(dct['neg_llhs']) for dct in detector.standardization_info])
neg_mx = np.max([np.max(dct['neg_llhs']) for dct in detector.standardization_info])
pos_mn = np.min([np.min(dct['pos_llhs']) for dct in detector.standardization_info])
pos_mx = np.max([np.max(dct['pos_llhs']) for dct in detector.standardization_info])

print neg_mn, pos_mn
if adjusted:
    mn = -300
    mx = 300
    dt = 10
else:
    mn = min(neg_mn, pos_mn) 
    mx = max(neg_mx, pos_mx)
    dt = 100

bins = np.arange(mn, mx, dt)

bkg_mx = np.max([np.max(sbkg) for sbkg in detector.fixed_spread_bkg])

x0 = np.linspace(mn, mx, 100)

if 0:
    def pdf(x, loc=0.0, scale=1.0):
        return np.exp(-(x - loc)**2 / (2*scale**2)) / np.sqrt(2*np.pi) / scale

    def logpdf(x, loc=0.0, scale=1.0):
        return -(x - loc)**2 / (2*scale**2) - 0.5 * np.log(2*np.pi) - np.log(scale)

fig = plt.figure(figsize=(13, 9))

# Limit the number viewed
offset = 0
#L = 3

from gv.fast import nonparametric_rescore 

for i in xrange(offset, offset+L):
    dct = detector.standardization_info[i]
    plt.subplot(L, 2, 1 + 2*(i-offset))
    print i
    print dct
    print '#neg', len(dct['neg_llhs'])
    print '#pos', len(dct['pos_llhs'])
    print 'unique negs', len(np.unique(dct['neg_llhs']))
    print '------------'
    #import pdb; pdb.set_trace()

    if adjusted:
        neg_R = dct['neg_llhs'].copy().reshape((-1, 1))
        pos_R = dct['pos_llhs'].copy().reshape((-1, 1))
        nonparametric_rescore(neg_R, dct['start'], dct['step'], dct['points'])
        nonparametric_rescore(pos_R, dct['start'], dct['step'], dct['points'])
        plt.hist(neg_R, alpha=0.5, label='neg', bins=bins, normed=True)
        plt.hist(pos_R, alpha=0.5, label='pos', bins=bins, normed=True)

    else:
        plt.hist(dct['neg_llhs'], alpha=0.5, label='neg', bins=bins, normed=True)
        plt.hist(dct['pos_llhs'], alpha=0.5, label='pos', bins=bins, normed=True)
   
    neg_N = len(dct['neg_llhs'])
    pos_N = len(dct['pos_llhs'])

    if 0:
        mu1 = np.mean(dct['neg_llhs'])
        mu2 = np.mean(dct['pos_llhs'])

        y = np.zeros_like(x0)

        neg_y = np.zeros_like(x0)
        for llh in dct['neg_llhs']:
            neg_y += pdf(x0, loc=llh, scale=50) / neg_N# * 0.99 + 0.01 * pdf(x0, loc=mu1, scale=100) / neg_N

        pos_y = np.zeros_like(x0)
        for llh in dct['pos_llhs']:
            pos_y += pdf(x0, loc=llh, scale=50) / pos_N# * 0.99 + 0.01 * pdf(x0, loc=mu2, scale=100) / pos_N

        neg_logs = np.zeros_like(dct['neg_llhs'])
        pos_logs = np.zeros_like(dct['pos_llhs'])

        #neg_hist = np.histogram(neg_logs)
        #pos_hist = np.histogram(pos_logs)

        neg_hist = np.histogram(dct['neg_llhs'], bins=10, normed=True)
        pos_hist = np.histogram(dct['pos_llhs'], bins=10, normed=True)

        def score2(R, neg_hist, pos_hist):
            neg_N = 0
            for j, weight in enumerate(neg_hist[0]):
                #import pdb; pdb.set_trace()
                if weight > 0:
                    llh = (neg_hist[1][j+1] + neg_hist[1][j]) / 2
                    neg_logs[neg_N] = np.log(weight) + logpdf(R, loc=llh, scale=200)
                    neg_N += 1

            pos_N = 0
            for j, weight in enumerate(pos_hist[0]):
                if weight > 0:
                    llh = (pos_hist[1][j+1] + pos_hist[1][j]) / 2
                    pos_logs[pos_N] = np.log(weight) + logpdf(R, loc=llh, scale=200)
                    pos_N += 1

            from scipy.misc import logsumexp
            return logsumexp(pos_logs[:pos_N]) - logsumexp(neg_logs[:neg_N])
        
        def score(R, neg_llhs, pos_llhs):
            for j, llh in enumerate(neg_llhs):
                neg_logs[j] = logpdf(R, loc=llh, scale=200)

            for j, llh in enumerate(pos_llhs):
                pos_logs[j] = logpdf(R, loc=llh, scale=200)

            from scipy.misc import logsumexp
            return logsumexp(pos_logs) - logsumexp(neg_logs)

        y2 = np.zeros_like(y)

        for k in xrange(len(x0)):
            #for j, llh in enumerate(dct['neg_llhs']):
                #neg_logs[j] = logpdf(x0[k], loc=llh, scale=200)
    #
            #for j, llh in enumerate(dct['pos_llhs']):
                #pos_logs[j] = logpdf(x0[k], loc=llh, scale=200)

            #y[k] = logsumexp(pos_logs) - logsumexp(neg_logs)
            #y[k] = score(x0[k], dct['neg_llhs'], dct['pos_llhs'])
            y[k] = score2(x0[k], neg_hist, pos_hist)
            y2[k] = score(x0[k], dct['neg_llhs'], dct['pos_llhs'])

    #plt.plot(x0, neg_y, linewidth=2.0, color='blue')
    #plt.plot(x0, pos_y, linewidth=2.0, color='green')
    if 0:
        #plt.plot(
        plt.twinx()
        #plt.plot(x0, pos_y / (pos_y + neg_y), linewidth=2.0, color='red')
        eps = 1e-290
        eps2 = eps * 3
        #import pdb; pdb.set_trace()
        sigma = 100.0
    #pos_y = np.maximum(neg_y, pos_y)
    #plt.plot(x0, np.log(pos_y + eps2) - np.log(neg_y + eps), linewidth=2.0, color='red')
    if not adjusted:
        plt.twinx()
        info = detector.standardization_info[i]
        x0 = np.asarray([info['start'] + info['step'] * k for k in xrange(len(info['points']))])
        y = info['points']
        plt.plot(x0, y, linewidth=2.0, color='red')
        #plt.plot(x0, y2, linewidth=1.0, color='blue')

    plt.xlim((mn, mx))
    #plt.ylim((0, 0.04))
    if i == L-1:
        if adjusted:
            label = 'score'
        else:
            label = 'LLH (without const.)'
        plt.xlabel(label)

    plt.subplot(L, 2, 2 + 2*(i-offset))
    if i == 0:
        plt.title('Background model')
    plt.plot(np.apply_over_axes(np.mean, detector.fixed_spread_bkg[i], [0, 1]).ravel())
    plt.ylim((0, bkg_mx))
    if i == L-1:
        plt.xlabel('Part #')

plt.legend()
plt.show()
