from __future__ import division
from __future__ import print_function
import argparse

parser = argparse.ArgumentParser(description='Test response of model')
parser.add_argument('results', metavar='<file>', nargs='+', type=argparse.FileType('rb'), help='Filename of results file')
parser.add_argument('-o', '--output', default=None, type=argparse.FileType('wb'), help='Filename of output file')
parser.add_argument('--title', default=None, type=str)
parser.add_argument('--captions', metavar='<caption>', nargs='*', type=str, help='Captions')
parser.add_argument('-b', '--background', default=None, type=argparse.FileType('rb'), help='Overlay in the background')

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
import os

def format_label(label, ap):
    return "{label} ({ap:.01f}%)".format(ap=100*ap, label=label)

use_other_style = False 

figsize = (8, 8)
if args.background is not None:
    backim = gv.img.load_image(args.background)
    figsize = (11, 8)
    
fig = plt.figure(figsize=figsize)
ax = fig.add_subplot(111)

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

    caption = format_label(caption, ap)

    extra = {}

    if args.background is not None:
        ax.imshow(backim, extent=(0, 1, 0, 1))
        if len(results_files) == 1:
            #extra['color'] = 'red' 
            extra['color'] = (0.0, 0.5, 0.0)
            extra['linewidth'] = 2
            extra['linestyle'] = 'solid'
            #extra['linestyle'] = 'dashed'

    if use_other_style:
        ax.plot(1-p, r, '-',label=caption, **extra)
        ax.set_xlabel('1-Precision')
        ax.set_ylabel('Recall')
    else:
        ax.plot(r, p, '-',label=caption, **extra)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))

if len(results_files) > 1:
    ax.legend()#fontsize='small')#, framealpha=0.2)

if args.background is None:
    ax.grid()
else:
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.65, box.height])

    if 'car' in args.background.name:
        data = [('Oxford', 0.432, 'black', 'solid'),
                ('UoCTTI', 0.346, 'blue', 'solid'),
                ('IRISA',  0.318, 'red', 'solid'),
                ('Damstadt', 0.301, 'lightgreen', 'solid'),
                ('INRIA_PlusClass', 0.294, 'Fuchsia', 'solid'),
                ('INRIA_Normal', 0.265, 'cyan', 'solid'),
                ('TKK', 0.184, 'yellow', 'solid'),
                ('MPI_Center', 0.172, 'black', 'dashed'),
                ('MPI_ESSOL', 0.120, 'blue', 'dashed'),]
    elif 'bicycle' in args.background.name:
        data = [('Oxford', 0.409, 'black', 'solid'),
                ('UoCTTI', 0.369, 'blue', 'solid'),
                ('INRIA_PlusClass', 0.287, 'red', 'solid'),
                ('IRISA',  0.281, 'lightgreen', 'solid'),
                ('INRIA_Normal', 0.246, 'Fuchsia', 'solid'),
                ('MPI_ESSOL', 0.157, 'cyan', 'solid'),
                ('MPI_Center', 0.110, 'yellow', 'solid'),
                ('TKK', 0.078, 'black', 'dashed'),]
    elif 'motorbike' in args.background.name:
        data = [('Oxford', 0.375, 'black', 'solid'),
                ('UoCTTI', 0.276, 'blue', 'solid'),
                ('INRIA_PlusClass', 0.249, 'red', 'solid'),
                ('IRISA',  0.227, 'lightgreen', 'solid'),
                ('MPI_ESSOL', 0.208, 'Fuchsia', 'solid'),
                ('MPI_Center', 0.170, 'cyan', 'solid'),
                ('INRIA_Normal', 0.153, 'yellow', 'solid'),
                ('TKK', 0.135, 'black', 'dashed'),]

    #labels = [d[0] for d in data]
    #scores = [d[1] for d in data]
    #coors = [d[2] for d in data]

    for label, score, color, linestyle in data:
        ax.plot([0], [0], label=format_label(label, score), color=color, linestyle=linestyle)

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

if args.title is not None:
    ax.set_title(args.title)
#plt.xticks(np.arange(0, 1+0.001, 0.05))
#plt.yticks(np.arange(0, 1+0.001, 0.05))
if args.output is not None:
    plt.savefig(args.output.name)
    os.chmod(args.output.name, 0644)
else:
    plt.show()

