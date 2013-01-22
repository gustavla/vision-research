
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

plt.hist(negs, 30, alpha=0.5, normed=True)
plt.hist(poss, 15, alpha=0.5, normed=True)

plt.show()

