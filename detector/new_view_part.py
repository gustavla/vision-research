
import argparse

parser = argparse.ArgumentParser(description='Train mixture model on edge data')
parser.add_argument('parts', metavar='<parts file>', type=argparse.FileType('rb'), help='Filename of parts file')
parser.add_argument('-i', dest='inspect', nargs=1, default=[None], metavar='INDEX', type=int, help='Run and inspect a single part')
parser.add_argument('--visparts', action='store_true', help='Visparts style')
parser.add_argument('--old', action='store_true', help='Old style')


args = parser.parse_args()
parts_file = args.parts
inspect_component = args.inspect[0]
use_visparts = args.visparts
use_old = args.old

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
    
    if use_old:
        p = np.rollaxis(parts_descriptor.parts[inspect_component], 2)
        ag.plot.images(p)

    elif use_visparts:
        p = parts_descriptor.visparts[inspect_component]
        ag.plot.images([p], zero_to_one=False)
    else:
        size = 29 
        
        if 1:
            p = parts[inspect_component]
            p = np.expand_dims(p, -1)
            print p.shape
            ag.plot.visualize_hog_color(p**4, (size, size), polarity_sensitive=False, phase=np.pi/2, direction=-1)
        else:
            p = parts[inspect_component]

            #p = p[:2,:2]

            p = np.rollaxis(p, 2)
            ag.plot.images(p)
