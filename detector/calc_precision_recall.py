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
#import matplotlib.pylab as plt
import gv

use_other_style = False 

all_ap = []

TYPE = '1-AP'
def transf(x):
    return 100*(1 - x)
def transf0(x):
    return 100*x

for i, results_file in enumerate(results_files):
    try:
        data = np.load(results_file)
    except:
        print("{} skipped".format(results_file.name))
        continue

    detections = data['detections']
    tp_fn = int(data['tp_fn'])
    p, r = gv.rescalc.calc_precision_recall(detections, tp_fn)

    ap = gv.rescalc.calc_ap(p, r) 
    all_ap.append(ap)

    N = 40
    n1 = int(ap*N)
    n2 = N - n1 
    bar = '#'*n1 + ' '*n2

    print('{name:60s} {bar} {t}: {ap:.02f}%'.format(ap=transf(ap), name=results_file.name, t=TYPE, bar=bar))
    #print(detections[-10:])

    if 0:
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

print('---------------')
print('{t} avg={avg:.02f}, std={std:.02f}, median={median:.02f}, min={min:.02f}, max={max:.02f}'.format(t=TYPE,
                                                                                 avg=transf(np.mean(all_ap)), 
                                                                                 std=transf0(np.std(all_ap, ddof=1)),
                                                                                 median=transf(np.median(all_ap)), 
                                                                                 min=transf(np.min(all_ap)),
                                                                                 max=transf(np.max(all_ap))))

print('---------------')
print(all_ap, ',')

if 0:
    plt.legend(fontsize='xx-small', framealpha=0.2)

    plt.grid()
    #plt.xticks(np.arange(0, 1+0.001, 0.05))
    #plt.yticks(np.arange(0, 1+0.001, 0.05))
    plt.show()

