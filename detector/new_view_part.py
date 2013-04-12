
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

parts_descriptor = gv.BinaryDescriptor.getclass('parts').load(parts_file)

originals = parts_descriptor.visparts
parts = parts_descriptor.parts

if inspect_component is not None:
    #p = np.rollaxis(parts[inspect_component], axis=2)

    if 1:
        p = parts[inspect_component]
        p = np.expand_dims(p, -1)
        print p.shape
        ag.plot.visualize_hog(p**4, (19, 19), polarity_sensitive=False)
    else:
        p = parts[inspect_component]

        #p = p[:2,:2]

        p = np.rollaxis(p, 2)
        ag.plot.images(p)
