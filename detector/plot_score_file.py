
import argparse

parser = argparse.ArgumentParser(description='Train mixture model on edge data')
parser.add_argument('score', metavar='<score file>', type=argparse.FileType('rb'), help='Score file')
parser.add_argument('--normalize', action='store_true', help='Normalize x axis')

args = parser.parse_args()
score_file = args.score
normalize = args.normalize

import numpy as np
import matplotlib.pylab as plt

obj = np.load(score_file)
d = obj.flat[0]

negs = d['llhs_negatives']
poss = d['llhs_positives']

alls = np.r_[negs, poss]


if normalize:
    mean, std = alls.mean(), alls.std()

    negs = (negs - mean) / std
    poss = (poss - mean) / std


mn, mx = min(negs.min(), poss.min()), max(negs.max(), poss.max())

if normalize:
    dbin = 0.2
else:
    dbin = 100

dbin = 0.5

mn = dbin * (mn//dbin) - dbin * 2 
mx = dbin * (mx//dbin) + dbin * 3

bins = np.arange(mn, mx+1e-10, dbin)

#bins = (bins - mean) / std

print 'Score: ', poss.mean() - negs.mean()

plt.hist(negs, bins, alpha=0.5, normed=True, label='Negatives')
plt.hist(poss, bins, alpha=0.5, normed=True, label='Positives')
plt.xlabel("Standardized log likelihood")
plt.ylabel("Normalized histogram")
plt.legend()

plt.show()

