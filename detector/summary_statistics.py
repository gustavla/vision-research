import argparse
from settings import load_settings

parser = argparse.ArgumentParser(description='Train mixture model on edge data')
parser.add_argument('settings', metavar='<settings file>', type=argparse.FileType('r'), help='Filename of settings file')
#parser.add_argument('--spread', action='store_true')
parser.add_argument('--limit', type=int, default=10)
parser.add_argument('--factor', type=float, default=1.0)
parser.add_argument('--shuffle', action='store_true')

args = parser.parse_args()
settings_file = args.settings
limit = args.limit
factor = args.factor
do_shuffle = args.shuffle

import gv
import os
import os.path
import glob
import numpy as np
import amitgroup as ag
from scipy.stats import mstats

settings = load_settings(settings_file)
descriptor = gv.load_descriptor(settings)

#parts_descriptor = gv.BinaryDescriptor.getclass('parts').load(parts_file)

path = os.path.expandvars(settings['detector']['neg_dir'])
files = sorted(glob.glob(path))

if do_shuffle:
    import random
    random.shuffle(files)

files = files[:limit]

pi = np.zeros(descriptor.num_parts)

tot = 0
cut = 4

intensities = np.array([])

num_e_count = None 
e_count = 0
e_tot = 0

radii = settings['detector']['spread_radii']
psize = settings['detector']['subsample_size']

for f in files:
    print 'Processing', f
    im = gv.img.load_image(f)
    im = gv.img.asgray(im)
    im = gv.img.resize_with_factor_new(im, factor)
    
    #intensities = np.concatenate((intensities, im.ravel())) 

    edges = ag.features.bedges(im, **descriptor.bedges_settings())

    feats = descriptor.extract_features(im, settings=dict(spread_radii=radii, subsample_size=psize, crop_border=cut))
    x = np.rollaxis(feats, 2).reshape((descriptor.num_parts, -1))
    tot += x.shape[1]
    pi += x.sum(axis=1)

    flat = edges.reshape((-1, edges.shape[-1]))
    e_count += flat.sum(axis=0)
    e_tot += np.prod(edges.shape[:-1])  

    es = flat.sum(axis=1)
    if num_e_count is None:
        num_e_count = es
    else:
        num_e_count = np.r_[num_e_count, es]

#if 1:
    #import pylab as plt
    #plt.hist(np.log(np.clip(intensities, 1e-4, 1-1e-4))/np.log(10), 50)
    #plt.show()
    
bkg = pi / tot
edges = e_count.astype(np.float64) / e_tot
avg_edges_per_pixel = num_e_count / e_tot
np.set_printoptions(precision=2)
print 'num_edges', np.bincount(num_e_count.astype(int), minlength=4) / float(num_e_count.size)
print 'edges', '{0:.2f}'.format(edges.mean()), edges
print 'bkg-avg', '{0:.3f}'.format(bkg.mean())
print 'bkg-min', '{0:.2f}'.format(bkg.min())
print 'bkg-max', '{0:.2f}'.format(bkg.max())
print 'quintiles', mstats.mquantiles(bkg, np.linspace(0, 1, 5))

if 0:
    print 'most common', np.argmax(bkg)
    import pylab as plt
    plt.plot(np.sort(bkg))
    plt.show()

if 0:
    if do_spreading:
        np.save('spread_bkg.npy', bkg)
    else:
        np.save('bkg.npy', bkg)
        
