from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='Plot llhs')
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')
parser.add_argument('--mixcomp', type=int, default=None)
parser.add_argument('--adjusted', action='store_true')
parser.add_argument('-o', '--output', type=str, default=None)

args = parser.parse_args()
model_file = args.model
adjusted = args.adjusted
mixcomp = args.mixcomp
output = args.output

import numpy as np
import gv
import matplotlib.pylab as plt

detector = gv.Detector.load(model_file)


if mixcomp is None:
    sinfo = [info[0] for info in detector.standardization_info]
else:
    sinfo = detector.standardization_info[mixcomp]

L = len(sinfo)

neg_mn = np.min([np.min(dct['neg_llhs']) for dct in sinfo])
neg_mx = np.max([np.max(dct['neg_llhs']) for dct in sinfo])
pos_mn = np.min([np.min(dct['pos_llhs']) for dct in sinfo])
pos_mx = np.max([np.max(dct['pos_llhs']) for dct in sinfo])

#print neg_mn, pos_mn

if adjusted:
    mn = -50
    mx = 50
    dt = 2.5

    mn = -1
    mx = 1
    dt = 0.05
else:
    mn = min(neg_mn, pos_mn) 
    mx = max(neg_mx, pos_mx)
    dt = 100
    
    #dt = 2.5 

bins = np.arange(mn, mx, dt)

bkg_mx = np.max([np.max(sbkg) for sbkg in detector.fixed_spread_bkg])

x0 = np.linspace(mn, mx, 100)

fig = plt.figure(figsize=(13, 9))

# Limit the number viewed
offset = 0

from gv.fast import nonparametric_rescore 

ax0 = None
for i in xrange(offset, offset+L):
    dct = sinfo[i]
    if i == 0:
        ax0 = plt.subplot(L, 2, 1 + 2*(i-offset))
    else:
        ax = plt.subplot(L, 2, 1 + 2*(i-offset), sharex=ax0) 

    #print i
    #print dct
    #print '#neg', len(dct['neg_llhs'])
    #print '#pos', len(dct['pos_llhs'])
    #print 'unique negs', len(np.unique(dct['neg_llhs']))
    #print '------------'
    #import pdb; pdb.set_trace()

    if adjusted:
        neg_R = dct['neg_llhs'].copy().reshape((-1, 1))
        pos_R = dct['pos_llhs'].copy().reshape((-1, 1))
        nonparametric_rescore(neg_R, dct['start'], dct['step'], dct['points'])
        nonparametric_rescore(pos_R, dct['start'], dct['step'], dct['points'])
        #neg_R = (neg_R - dct['neg_llhs'].mean()) / dct['neg_llhs'].std()
        #pos_R = (pos_R - dct['neg_llhs'].mean()) / dct['neg_llhs'].std()
        #neg_R = (neg_R - dct['pos_llhs'].mean()) / dct['pos_llhs'].std()
        #pos_R = (pos_R - dct['pos_llhs'].mean()) / dct['pos_llhs'].std()
        plt.hist(neg_R, alpha=0.5, label='neg', bins=bins, normed=True)
        plt.hist(pos_R, alpha=0.5, label='pos', bins=bins, normed=True)

    else:
        plt.hist(dct['neg_llhs'], alpha=0.5, label='neg', bins=bins, normed=True)
        plt.hist(dct['pos_llhs'], alpha=0.5, label='pos', bins=bins, normed=True)
   
    neg_N = len(dct['neg_llhs'])
    pos_N = len(dct['pos_llhs'])

    if not adjusted:
        print 'Plotting', i
        plt.twinx()
        info = sinfo[i]
        #x0 = np.asarray([info['start'] + info['step'] * k for k in xrange(len(info['points']))])
        y = info['points']
        x0 = np.arange(y.size) * info['step'] + info['start']
        plt.plot(x0, y, linewidth=2.0, color='red')
        print x0[0], x0[-1], y[0], y[-1]
        #plt.plot(x0, y2, linewidth=1.0, color='blue')

    plt.xlim((mn, mx))
    #plt.ylim((-30, 30))
    if i == L-1:
        if adjusted:
            label = 'score'
        else:
            label = 'LLH (without const.)'
        plt.xlabel(label)

    plt.subplot(L, 2, 2 + 2*(i-offset))
    #if i == 0:
        #plt.title('Background model')
    #plt.plot(np.apply_over_axes(np.mean, detector.fixed_spread_bkg[i], [0, 1]).ravel())
    if mixcomp is None:
        bi = 0 
    else:
        bi = i 
    plt.plot(detector.bkg_mixture_params[bi])
    plt.ylim((0, bkg_mx))
    if i == L-1:
        plt.xlabel('Part #')

#plt.subplot(L, 2, 1)
#plt.xlim((mn, mx))

#plt.legend()
if output:
    plt.savefig(output)
else:
    plt.show()
