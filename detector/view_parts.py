
import argparse

parser = argparse.ArgumentParser(description='Train mixture model on edge data')
parser.add_argument('parts', metavar='<parts file>', type=argparse.FileType('rb'), help='Filename of parts file')
parser.add_argument('-i', dest='inspect', nargs=1, default=[None], metavar='INDEX', type=int, help='Run and inspect a single part')
parser.add_argument('--plot-prevalence', action='store_true', help='Plot the prevalence of each part')

args = parser.parse_args()
parts_file = args.parts
inspect_component = args.inspect[0]
plot_prevalence = args.plot_prevalence

import numpy as np
import matplotlib.pylab as plt
import amitgroup as ag
import amitgroup.features
import sys
import gv


#parts_dictionary = gv.PatchDictionary.load(part_file)
parts_descriptor = gv.BinaryDescriptor.getclass('parts').load(parts_file)

originals = parts_descriptor.visparts
parts = parts_descriptor.parts

if inspect_component is not None:
    #if inspect_component == 0:
    #    print "Can't plot background part"
    #    sys.exit(1)
    #ag.plot.images
    p = np.rollaxis(parts[inspect_component], axis=2)
    print "{5} Min/max/avg/std/median probabilities: {0:.2f} {1:.2f} {2:.2f} {3:.2f} {4:.2f}".format(p.min(), p.max(), p.mean(), p.std(), np.median(p), inspect_component)
    ag.plot.images(p)
else:
    if plot_prevalence:
        arr = []
        for inspect_component in xrange(100):
            p = np.rollaxis(parts[inspect_component], axis=2)
            print "{5} Min/max/avg/std/median probabilities: {0:.2f} {1:.2f} {2:.2f} {3:.2f} {4:.2f}".format(p.min(), p.max(), p.mean(), p.std(), np.median(p), inspect_component)

            arr.append([p.min(), p.max(), p.mean(), p.std(), np.median(p)])
        arr = np.asarray(arr)
        import matplotlib.pylab as plt
        plt.plot(arr[...,0])
        plt.show()
    else:
        ag.plot.images(originals, zero_to_one=False)

#ag.plot.images(np.rollaxis(parts[9], axis=2))
