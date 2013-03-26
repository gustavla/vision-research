import argparse

parser = argparse.ArgumentParser(description='Train mixture model on edge data')
parser.add_argument('parts', metavar='<parts file>', type=argparse.FileType('rb'), help='Filename of parts file')

args = parser.parse_args()
parts_file = args.parts

import gv
import os
import os.path
import glob
import numpy as np

parts_descriptor = gv.BinaryDescriptor.getclass('parts').load(parts_file)

path = os.path.join(os.environ['UIUC_DIR'], 'TrainImages/neg-*.pgm')
files = glob.glob(path)

pi = np.zeros(parts_descriptor.num_parts)

tot = 0

for f in files:
    print 'Processing', f
    im = gv.img.load_image(f)

    feats = parts_descriptor.extract_features(im, settings=dict(spread_radii=(2, 2), subsample_size=(2, 2)))
    x = np.rollaxis(feats, 2).reshape((parts_descriptor.num_parts, -1))
    tot += x.shape[1]
    pi += x.sum(axis=1)
    
bkg = pi / tot
print bkg.shape
np.save('bkg.npy', bkg)
