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

import matplotlib
if args.output is not None:
    matplotlib.use('Agg')
import numpy as np
import matplotlib.pylab as plt
import gv
import os

for i, results_file in enumerate(results_files):
    data = np.load(results_file)

    miss_rates = data['miss_rates']
    fppws = data['fppws']

    #print(results_file.name)
    #print('AP: {0:.02f}% ({1})'.format(100*ap, ap))
    #print(detections[-10:])
    #print()

    if captions:
        caption = captions[i]
    else:
        caption = 'File ' + results_file.name

    #print('caption', caption)
    caption = "{1}".format(0, caption)
    #caption = "{}".format(caption) 

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.plot(fppws, miss_rates, '-', label=caption)
    ax.set_title('Detection error tradeoff (DET)')
    ax.set_xlabel('False positives per window (FPPW)')
    ax.set_ylabel('Miss rate')
    ax.set_yticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.xlim((10**-6, 10**-1))
    plt.ylim((0.01, 0.5))

plt.legend()#fontsize='small')#, framealpha=0.2)

plt.grid()
#plt.xticks(np.arange(0, 1+0.001, 0.05))
#plt.yticks(np.arange(0, 1+0.001, 0.05))
if args.output is not None:
    plt.savefig(args.output)
    os.chmod(args.output.name, 0644)
else:
    plt.show()

