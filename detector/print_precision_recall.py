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
import gv

use_other_style = False 

aps = []

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

    aps.append(ap)

print(aps)

