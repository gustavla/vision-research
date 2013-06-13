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
import amitgroup as ag

parts_descriptor = gv.BinaryDescriptor.getclass('parts').load(parts_file)

path = os.path.join(os.environ['UIUC_DIR'], 'TestImages/test-*.pgm')
#path = os.path.join(os.environ['VOC_DIR'], 'JPEGImages/*.jpg')  
files = sorted(glob.glob(path))[:40]

pi = np.zeros(parts_descriptor.num_parts)

tot = 0
cut = 4

intensities = np.array([])

if do_spreading:
    # TODO: Should be specified
    radii = (2, 2)
else:
    radii = (0, 0)

e_count = 0
e_tot = 0

for f in files:
    print 'Processing', f
    im = gv.img.load_image(f)
    gray_img = gv.img.asgray(im)
    intensities = np.concatenate((intensities, gray_img.ravel())) 

    edges = ag.features.bedges(gray_img, **parts_descriptor.bedges_settings())
    feats = parts_descriptor.extract_features(gray_img, settings=dict(spread_radii=radii, subsample_size=(1, 1)))
    x = np.rollaxis(feats[cut:-cut,cut:-cut], 2).reshape((parts_descriptor.num_parts, -1))
    tot += x.shape[1]
    pi += x.sum(axis=1)

    e_count += edges.sum()
    e_tot += np.prod(edges.shape)  

if 1:
    import pylab as plt
    plt.hist(np.log(np.clip(intensities, 1e-4, 1-1e-4))/np.log(10), 50)
    plt.show()
    
bkg = pi / tot
print 'edges', e_count / e_tot
print bkg.shape
if do_spreading:
    np.save('spread_bkg.npy', bkg)
else:
    np.save('bkg.npy', bkg)
    
