from __future__ import division
from __future__ import print_function
import argparse

parser = argparse.ArgumentParser(description='Test response of model')
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')
parser.add_argument('results', metavar='<file>', nargs=1, type=argparse.FileType('rb'), help='Filename of results file')

args = parser.parse_args()
model_file = args.model
results_file = args.results[0]

import numpy as np
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
import gv

detector = gv.Detector.load(model_file)
data = np.load(results_file)

#p, r = data['precisions'], data['recalls']
detections = data['detections']
mixcomps = detections['mixcomp'].max() + 1

#bins = np.arange(0.0, 0.2, 0.01)
bins = np.arange(-20, 20, 0.25)

tps_fps = np.zeros((mixcomps, 2))

fig = plt.figure(figsize=(15, 7))

#gs = gridspec.GridSpec(mixcomps, 2, width_ratios=[7,1])
gs = gridspec.GridSpec(mixcomps, 10)
for mixcomp in xrange(mixcomps):
    mydets = detections[detections['mixcomp'] == mixcomp]

    tps_fps[mixcomp,0] = mydets[mydets['correct'] == 0].size
    tps_fps[mixcomp,1] = mydets[mydets['correct'] == 1].size

    ax = plt.subplot(gs[mixcomps-1-mixcomp,:5])
    #plt.subplot(gs[2*mixcomp])
    plt.hist(mydets[mydets['correct'] == 0]['confidence'], label='FP', bins=bins, alpha=0.5, normed=True)
    print(mixcomp, "mean:", np.mean(mydets[mydets['correct'] == 0]['confidence']))
    if len(mydets[mydets['correct'] == 1]):
        plt.hist(mydets[mydets['correct'] == 1]['confidence'], label='TP', bins=bins, alpha=0.5, normed=True)
    plt.ylim((0, 0.5))
    #plt.ylim((0, 25))

    if mixcomp == 0:
        plt.xlabel('Confidence (llh)')
    if mixcomp == mixcomps-1:
        plt.title('Confidence histograms')

    #plt.plot(r, p, label=str(mixcomp))
    #plt.xlabel('Recall')
    #plt.ylabel('Precision')
    #plt.xlim((0, 1))
    #plt.ylim((0, 1))

    #plt.subplot(gs[2*mixcomp+1]).set_axis_off()
    ax = plt.subplot(gs[mixcomps-1-mixcomp,5]).set_axis_off()
    l = plt.imshow(detector.support[mixcomp], cmap=plt.cm.gray)

# Plot which one

plt.legend()
#plt.show()

ind = np.arange(mixcomps)

width = 0.35
ax = plt.subplot(gs[:,6:])
ax.barh(ind, tps_fps[:,0], height=width, color='b', alpha=0.5, label='FP')
ax.barh(ind+width, tps_fps[:,1], height=width, color='g', alpha=0.5, label='TP')
plt.xlabel('Count')
plt.title('TP/FP ratios')
plt.legend()
plt.show()

