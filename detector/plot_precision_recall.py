from __future__ import division
from __future__ import print_function
import argparse

parser = argparse.ArgumentParser(description='Test response of model')
parser.add_argument('results', metavar='<file>', nargs='+', type=argparse.FileType('rb'), help='Filename of results file')
parser.add_argument('-o', '--output', default=None, type=argparse.FileType('wb'), help='Filename of output file')
parser.add_argument('--captions', metavar='<caption>', nargs='*', type=str, help='Captions')

args = parser.parse_args()
results_files = args.results
captions = args.captions
if captions:
    assert len(captions) == len(results_files), "Must supply caption for all"

if args.output is not None:
    import matplotlib
    matplotlib.use('Agg')
import numpy as np
import matplotlib.pylab as plt
import gv

use_other_style = False 


for i, results_file in enumerate(results_files):
    data = np.load(results_file)

    detections = data['detections']
    tp_fn = int(data['tp_fn'])
    p, r = gv.rescalc.calc_precision_recall(detections, tp_fn)

    ap = gv.rescalc.calc_ap(p, r) 

    print(results_file.name)
    print('AP: {0:.02f}% ({1})'.format(100*ap, ap))
    #print(detections[-10:])
    print()

    if captions:
        caption = captions[i]
    else:
        caption = results_file.name

    caption = "{0:.02f}% {1}".format(100*ap, caption)

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

plt.legend()#fontsize='small')#, framealpha=0.2)

plt.grid()
#plt.xticks(np.arange(0, 1+0.001, 0.05))
#plt.yticks(np.arange(0, 1+0.001, 0.05))
if args.output is not None:
    plt.savefig(args.output)
else:
    plt.show()

