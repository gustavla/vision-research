
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

mn, mx = min(negs.min(), poss.min()), max(negs.max(), poss.max())

mn = 100 * (mn//100) - 200
mx = 100 * (mx//100) + 300

bins = np.arange(mn, mx+1, 100)

plt.hist(negs, bins, alpha=0.5, normed=True)
plt.hist(poss, bins, alpha=0.5, normed=True)

plt.show()

