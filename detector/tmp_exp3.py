
from __future__ import division

import argparse

parser = argparse.ArgumentParser(description='Test response of model')
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')

args = parser.parse_args()
model_file = args.model

import numpy as np
import scipy.special 
import gv

detector = gv.Detector.load(model_file)

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

    counts = np.bincount(top_samples[1].astype(np.int64), minlength=len(amount))
    return counts

def gen_data(rs, amount, mu, sigma): 
    return rs.normal(mu, sigma, size=amount)     

np.set_printoptions(precision=2, suppress=True)

mixcomp = 0

info = detector.standardization_info[mixcomp][0]
neg_mu = info['neg_llhs'].mean()
neg_sg = info['neg_llhs'].std()
pos_mu = info['pos_llhs'].mean()
pos_sg = info['pos_llhs'].std()

pos_mu = 40

pos_mu = (pos_mu - neg_mu) / neg_sg
pos_sg = pos_sg / neg_sg
neg_mu = 0
neg_sg = 1

print 'Positive distribution'
print pos_mu, pos_sg

data = np.load('prev.npz')
xs, ys, zs = data['xs'], data['ys'], data['zs']

MUL = 1000

xs = xs[15:]
ys = ys[15:]
zs = zs[15:]
pixels = 100 * (500/2**xs)
feat_sizes = pixels / 4 # subsampling
feat_areas = (feat_sizes)**2

# Remove some due to correlation patterns
feat_areas /= feat_areas[-1]

nlevels = len(xs)
nsamples = np.round(feat_areas).astype(np.int64)

ALOT = nsamples.sum() * MUL

cur = 0 

import pandas as pd

aps = []
params = np.arange(0.4, 1.0, 0.05)
samples = pd.DataFrame(np.zeros((ALOT,),dtype=[('score', 'f8'),('correct', 'i4'),('scale', 'i4')]))

nimages = 5011
probs = ys / feat_areas / nimages

tt = np.round(1 / probs)#.astype(np.int64)

np.savez('tt.npz', xs=xs, tt=tt)

from scipy import stats as st

if 1:
    #mus = np.zeros(nlevels)
    #sigs = np.zeros(nlevels)

    #tt *= 0.95 

    mus = st.norm.ppf(1 - 1/tt)
    sigs = st.norm.ppf(1 - 1/tt/np.exp(1)) - mus
    #for i, lev in enumerate(xs):

    # Log ratios

    #import pdb; pdb.set_trace()


rs = np.random.RandomState(0)

for i, lev in enumerate(xs):
    npos = np.round(MUL * nsamples[i] * probs[i]).astype(np.int64)
    nneg = MUL * nsamples[i] - npos

    samples[cur:cur+nneg].score = rs.normal(neg_mu, neg_sg, size=nneg)
    samples[cur:cur+nneg].correct[:] = 0 
    samples[cur:cur+nneg].scale[:] = i
    cur += nneg
    samples[cur:cur+npos].score = rs.normal(pos_mu, pos_sg, size=npos)
    samples[cur:cur+npos].correct[:] = 1
    samples[cur:cur+npos].scale[:] = i
    cur += npos

    assert cur <= ALOT

#ss = samples[:cur]
ss = samples

print '-----'
print ss.score.min()

ss.score -= neg_mu
ss.score /= neg_sg

print ss.score.min()

#pos_mu += 10.0
print tt

params = [1]
for param in params: 
    ssi = ss.copy()

    for i, lev in enumerate(xs):
        from scipy import stats as st
        #ssi.score[ssi.scale == i] += 0.01 * param * 2**lev
        #x2 = np.log(2**(2*lev)) + st.norm.logpdf(lev, loc=7.5, scale=1.6)
        #alpha = 0.35
        #ssi.score[ssi.scale == i] += x2 * 0.45 * param #x2 * alpha + x1 * (1 - alpha) 

        s = ssi.score[ssi.scale == i]
        #s[:] = st.norm.logpdf(s, loc=pos_mu, scale=pos_sg) - \
               #st.genextreme.logpdf(s, 0, loc=mus[i], scale=sigs[i])

        if 1:
            #alpha = 1/(1 + nsamples[i])

            #st.norm.logpdf(
            #s[:] -= np.log(alpha / (1 - alpha)) / pos_mu
            #s[:] -= pos_mu**2/2

            pos_mu0 = pos_mu #+ 0.1
            print 'pos_mu0', pos_mu0
            
            #s -= np.log(nsamples[i]) / pos_mu0
            s -= np.log(tt[i]) / pos_mu0

            #s[:] += np.log(alpha / (1 - alpha))

        elif 1:
            EM = 0.57
            #real_mu = mus[i] + EM * sigs[i]
            #real_sig = sigs[i] * np.pi / np.sqrt(6.0)
            real_mu = mus[i]
            real_sig = sigs[i]
            s -= real_mu 
            #s /= real_sig
        elif 0:
            print i, 'pos_mu', pos_mu, 'mu', mus[i]
            #s2 = st.norm.logpdf(s, loc=pos_mu + 4.3) - \
            #       st.norm.logpdf(s, loc=mus[i])# - np.log(tt[i])

            if pos_mu > mus[i]:
                s[:] *= pos_mu - mus[i]
                s[:] -= pos_mu**2/2 - mus[i]**2/2
            else:
                pos_mu0 = mus[i] + 0.5
                s[:] *= pos_mu0 - mus[i]
                s[:] -= pos_mu0**2/2 - mus[i]**2/2
            #s[:] *= pos_mu
            #s[:] -= np.log(tt[i]) / pos_mu
            pass
        else:
            s += 0.01 * 0.75 * param * 2**lev
              
        ssi.score[ssi.scale == i] = s
        #ssi.score[ssi.scale == i] += st.norm.logpdf(

        


    #ths = np.linspace(ss.score.min(), ss.score.max(), 20)
    #print ths

    ssi = ssi.sort('score')
    p, r = gv.rescalc.calc_precision_recall(ssi, np.sum(ssi.correct == 1))
    ap = gv.rescalc.calc_ap(p, r) 
    print 'AP: {:0.2f}%'.format(ap*100)
    aps.append(ap)

    if 0:
        import pylab as plt
        plt.plot(r, p)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.show()


if 0:
    import pylab as plt
    plt.plot(params, aps)
    plt.show()




if 1:
    from scipy.stats import mstats
    def calc_precisions(ss_):
        return ((ss_[ss_.correct == 1]).groupby('scale').size().astype(np.float64) / ss_.groupby('scale').size())


    ths = mstats.mquantiles(np.asarray(ssi.score[ssi.correct == 1]), np.linspace(0.05, 0.95, 20))

    import pylab as plt
    for th in ths:
        pr = calc_precisions(ss[ss.score >= th])
        if xs.size == pr.size:
            plt.plot(xs, pr, label='{}'.format(th))

        #plt.plot( 

    plt.legend(fontsize='xx-small', framealpha=0.2)
    plt.show()

bins = np.linspace(-10, 10, 200)

