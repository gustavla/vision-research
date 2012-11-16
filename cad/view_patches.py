
import argparse

parser = argparse.ArgumentParser(description='Train mixture model on edge data')
parser.add_argument('patch', metavar='<patch file>', type=argparse.FileType('rb'), help='Filename of patches file')
parser.add_argument('-i', dest='inspect', nargs=1, default=[None], metavar='INDEX', type=int, help='Run and inspect a single patch')
parser.add_argument('--plot-prevalence', action='store_true', help='Plot the prevalence of each patch')

args = parser.parse_args()
patch_file = args.patch
inspect_component = args.inspect[0]
plot_prevalence = args.plot_prevalence

import numpy as np
import matplotlib.pylab as plt
import amitgroup as ag
import amitgroup.features
import sys

patch_data = np.load(patch_file)

originals = patch_data['vispatches']
patches = patch_data['patches']

if inspect_component is not None:
    if inspect_component == 0:
        print "Can't plot background patch"
        sys.exit(1)
    #ag.plot.images
    p = np.rollaxis(patches[inspect_component-1], axis=2)
    print "{5} Min/max/avg/std/median probabilities: {0:.2f} {1:.2f} {2:.2f} {3:.2f} {4:.2f}".format(p.min(), p.max(), p.mean(), p.std(), np.median(p), inspect_component)
    ag.plot.images(p)
else:
    if plot_prevalence:
        arr = []
        for inspect_component in xrange(100):
            p = np.rollaxis(patches[inspect_component], axis=2)
            print "{5} Min/max/avg/std/median probabilities: {0:.2f} {1:.2f} {2:.2f} {3:.2f} {4:.2f}".format(p.min(), p.max(), p.mean(), p.std(), np.median(p), inspect_component)

            arr.append([p.min(), p.max(), p.mean(), p.std(), np.median(p)])
        arr = np.asarray(arr)
        import matplotlib.pylab as plt
        plt.plot(arr[...,0])
        plt.show()
    else:
        ag.plot.images(originals/originals.max())

#ag.plot.images(np.rollaxis(patches[9], axis=2))
