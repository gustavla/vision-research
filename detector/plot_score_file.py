
import argparse

parser = argparse.ArgumentParser(description='Train mixture model on edge data')
parser.add_argument('score', metavar='<score file>', type=argparse.FileType('rb'), help='Score file')

args = parser.parse_args()
score_file = args.score

import numpy as np
import matplotlib.pylab as plt

obj = np.load(score_file)
d = obj.flat[0]

negs = d['llhs_negatives']
poss = d['llhs_positives']

alls = np.r_[negs, poss]

mean, std = alls.mean(), alls.std()

negs = (negs - mean) / std
poss = (poss - mean) / std

mn, mx = min(negs.min(), poss.min()), max(negs.max(), poss.max())

dbin = 0.2

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

