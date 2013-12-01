from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='Calculate FP rates')
parser.add_argument('results', metavar='<file>', nargs='+', type=argparse.FileType('rb'), help='Filename of results file')



args = parser.parse_args()
results_files = args.results

import numpy as np
import matplotlib.pylab as plt
import gv
import pandas as pd

np.set_printoptions(precision=2,suppress=True)

def calc_fp_rate(detections, tp_fn, factor, nlevels, offset):
    counts = np.zeros((2, nlevels))

    # Add scale index value to dataframe
    detections['scale_index'] = np.round(np.log2(detections['scale']) / factor - offset)

    for f in xrange(offset, offset+nlevels):
        dets = detections[detections.scale_index == f]
        for i in xrange(2):
             
            counts[i, f-offset] += (dets.correct == i).sum()

    if 0:
        for i, det in detections.iterrows():
            #import pdb; pdb.set_trace()
            lev = np.log2(det['scale']) / factor - offset
            print lev
            counts[det['correct'],lev] += 1 

    return counts


for results_file in results_files:
    data = np.load(results_file)
    detections = pd.DataFrame(data['detections'])
    tp_fn = int(data['tp_fn'])

    min_conf = detections.confidence.min()
    max_conf = detections.confidence.max()
    print 'min conf', min_conf 
    print 'max conf', max_conf 

    for th in np.linspace(min_conf, max_conf, 20):
        #th = 17.0

        # Get the scales
        log_scales = np.log2(np.sort(detections.scale.unique()))
        factor = np.diff(log_scales)[0]
        print factor
        if 0:
            plt.plot(log_scales)
            plt.show()
        levels = np.round(np.log2(np.sort(detections.scale.unique())) / factor).astype(np.int32)

        offset = levels[0]
        print levels

        # Number of possible detections (up to a constant) for a certain level
        pixels = 512**2 / 2**(2*levels)

        #print scales

        counts = calc_fp_rate(detections[detections.confidence >= th], tp_fn, factor, len(levels), offset)

        nimg = detections.img_id.unique().size
        print 'nimg', nimg

        print counts
        fp = counts[0].sum()
        tp = counts[1].sum()

        fprate = fp / (pixels * nimg)
        rel_fprate = fprate / fprate.sum()
        precisions = counts[1] / counts.sum(axis=0)
        print 'Precision', tp / (tp + fp)
        print 'Recall', tp / tp_fn
        print 'FP rate', rel_fprate
          
        plt.plot(levels, precisions * 100, label='{0} - {1}'.format(results_file.name, th))
#plt.ylim((0, 20))
plt.ylabel('Precision (%)')
plt.xticks(levels, ['{:0.1f}'.format(500/10*5/4*2**(lev*factor)) for lev in levels])
plt.xlabel('Scale level index')
plt.legend(loc=2, fontsize='xx-small', framealpha=0.2)
plt.show()
