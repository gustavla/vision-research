from __future__ import division
from __future__ import print_function
import argparse

parser = argparse.ArgumentParser(description='Test response of model')
parser.add_argument('results', metavar='<file>', nargs='+', type=argparse.FileType('rb'), help='Filename of results file')

args = parser.parse_args()
results_files = args.results

import numpy as np
import gv

use_other_style = False 

tots = []
for i, results_file in enumerate(results_files):
    try:
        data = np.load(results_file)
    except IOError:
        print(results_file.name, "[skipping]")
        continue
    
    try:
        num_images = data['num_images']
    except:
        num_images = 741
    detections = data['detections']
    tp_fn = int(data['tp_fn'])
    try:
        fppi, miss_rate = gv.rescalc.calc_fppi_miss_rate(detections, tp_fn, num_images)
        summary = gv.rescalc.calc_fppi_summary(fppi, miss_rate) 
    except IndexError:
        print(results_file.name, "[skipping]")
        continue


    tots.append(summary)


    N = 40
    n1 = int(summary*N)
    n2 = N - n1 
    bar = '#'*n1 + ' '*n2

    print('{name:80s} {bar}: {ap:.02f}%'.format(ap=summary * 100, name=results_file.name, bar=bar))

print('mean', np.mean(tots))
print(tots, ',')
