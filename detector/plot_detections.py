from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='Test response of model')
parser.add_argument('results', metavar='<file>', type=argparse.FileType('rb'), help='Filename of results file')
parser.add_argument('-o', '--output', type=argparse.FileType('wb'), help='Filename of output image')

args = parser.parse_args()
results_file = args.results

if args.output is not None:
    import matplotlib as mpl
    mpl.use('Agg')
import numpy as np
import matplotlib.pylab as plt

data = np.load(results_file)

#p, r = data['precisions'], data['recalls']
detections = data['detections']

#detections.sort(order='img_id')

N = len(detections)

x = np.zeros(N)
y = np.zeros(N)
c = ['g' for i in xrange(N)] 

max_N = detections['img_id'].astype(np.int).max()

all_negs = detections[detections['correct']==0]['confidence']

from scipy.stats import scoreatpercentile

th_score = scoreatpercentile(all_negs, 85)
detections = detections[detections['confidence'] >= th_score]

mx = detections['confidence'].max()
mx += (mx - th_score) * 0.05

print 'number of detections', len(detections)

#print detections['img_id']
#SHIFT = int(detections['img_id'][-1]) * 10
SHIFT = max_N/2
print SHIFT

NN = 20000
#MM = 2000

for i, d in enumerate(detections):
    x[i] = d['img_id'] 
    x[i] += NN * d['mixcomp']
    #x[i] += MM * d['bkgcomp']
    y[i] = d['confidence']
    c[i] = ['r', 'g'][d['correct']]

fig = plt.figure()
ax = fig.add_subplot(111)
M = detections['mixcomp'].max()+1
ax.set_xticks(np.arange(M) * NN + SHIFT)
ax.set_xticklabels(map(str, 1+np.arange(M)))
ax.set_ylim((th_score, mx))
ax.set_ylabel('Score')
ax.set_xlabel('Object component')
ax.scatter(x, y, c=c, lw=0, edgecolors='none', s=15, alpha=1.0)
if args.output:
    plt.savefig(args.output.name)
else:
    plt.show()
