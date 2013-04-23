import argparse

parser = argparse.ArgumentParser(description='Train mixture model on edge data')
parser.add_argument('parts', metavar='<parts file>', type=argparse.FileType('rb'), help='Filename of parts file')
parser.add_argument('--spread', action='store_true')

args = parser.parse_args()
parts_file = args.parts
do_spreading = args.spread

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
cut = 4

intensities = np.array([])

if do_spreading:
    # TODO: Should be specified
    radii = (2, 2)
else:
    radii = (0, 0)


for f in files:
    print 'Processing', f
    im = gv.img.load_image(f)
    gray_img = gv.img.asgray(im)
    intensities = np.concatenate((intensities, gray_img.ravel())) 

    feats = parts_descriptor.extract_features(im, settings=dict(spread_radii=radii, subsample_size=(1, 1)))
    x = np.rollaxis(feats[cut:-cut,cut:-cut], 2).reshape((parts_descriptor.num_parts, -1))
    tot += x.shape[1]
    pi += x.sum(axis=1)

if 0:
    import pylab as plt
    plt.hist(intensities, 50)
    plt.show()
    
bkg = pi / tot
print bkg.shape
if do_spreading:
    np.save('spread_bkg.npy', bkg)
else:
    np.save('bkg.npy', bkg)
    
