from __future__ import division
from __future__ import print_function
import argparse

parser = argparse.ArgumentParser(description='Test response of model')
parser.add_argument('results', metavar='<file>', nargs='+', type=argparse.FileType('rb'), help='Filename of results file')
parser.add_argument('--captions', metavar='<caption>', nargs='*', type=str, help='Captions')

args = parser.parse_args()
results_files = args.results
captions = args.captions
if captions:
    assert len(captions) == len(results_files), "Must supply caption for all"

import numpy as np
import scipy.integrate
import matplotlib.pylab as plt
import gv

use_other_style = True


for i, results_file in enumerate(results_files):
    data = np.load(results_file)

    detections = data['detections']
    tp_fn = int(data['tp_fn'])
    p, r = gv.rescalc.calc_precision_recall(detections, tp_fn)

    ap = gv.rescalc.calc_ap(p, r) 

    print(results_file.name)
    print('AP:', ap)
    #print(detections[-10:])
    print()

    if captions:
        caption = captions[i]
    else:
        caption = results_file.name

    if use_other_style:
        plt.plot(1-p, r, '-',label=caption)
        plt.xlabel('1-Precision')
        plt.ylabel('Recall')
    else:
        plt.plot(r, p, '-',label=caption)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
    plt.xlim((0, 1))
    plt.ylim((0, 1))

if use_other_style:
    plt.legend(loc=4)
else:
    plt.legend(loc=3)
plt.grid()
#plt.xticks(np.arange(0, 1+0.001, 0.05))
#plt.yticks(np.arange(0, 1+0.001, 0.05))
plt.show()
