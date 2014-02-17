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

fig = plt.figure(figsize=(7, 7))

for i, results_file in enumerate(results_files):
    data = np.load(results_file)
    
    try:
        num_images = data['num_images']
    except:
        num_images = 741
    detections = data['detections']
    tp_fn = int(data['tp_fn'])
    fppi, miss_rate = gv.rescalc.calc_fppi_miss_rate(detections, tp_fn, num_images)

    summary = gv.rescalc.calc_fppi_summary(fppi, miss_rate) 

    print(results_file.name)
    print('Avg miss rate: {0:.02f}% ({1})'.format(100*summary, summary))
    #print(detections[-10:])
    print()

    if captions:
        caption = captions[i]
    else:
        caption = results_file.name

    caption = "{0:.02f}% {1}".format(100*summary, caption)

    ax = fig.add_subplot(1, 1, 1)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.plot(fppi, miss_rate, '-',label=caption)
    ax.set_xlabel('FPPI')
    ax.set_ylabel('Miss rate')
    ticks = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax.set_yticks(ticks)
    ax.set_yticklabels(map(str, ticks))
    #plt.xlim((0, 1))
    #plt.ylim((0, 1))
    plt.xlim((3*10**-3, 10**0))
    plt.ylim((0.025, 1.0))

plt.legend(loc=3)#fontsize='small')#, framealpha=0.2)

plt.grid()
#plt.xticks(np.arange(0, 1+0.001, 0.05))
#plt.yticks(np.arange(0, 1+0.001, 0.05))
if args.output is not None:
    plt.savefig(args.output)
else:
    plt.show()

