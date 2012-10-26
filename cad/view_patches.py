
import argparse

parser = argparse.ArgumentParser(description='Train mixture model on edge data')
parser.add_argument('patch', metavar='<patch file>', type=argparse.FileType('rb'), help='Filename of patches file')

args = parser.parse_args()
patch_file = args.patch

import numpy as np
import matplotlib.pylab as plt
import amitgroup as ag
import amitgroup.features

patch_data = np.load(patch_file)

originals = patch_data['originals']
patches = patch_data['patches']

ag.plot.images(originals/originals.max())

#ag.plot.images(np.rollaxis(patches[9], axis=2))
