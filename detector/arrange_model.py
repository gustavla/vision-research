from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='Test response of model')
parser.add_argument('settings', metavar='<settings file>', type=argparse.FileType('r'), help='Filename of settings file')
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')
parser.add_argument('output', metavar='<output file>', type=argparse.FileType('wb'), help='Filename of output model file')

args = parser.parse_args()
settings_file = args.settings
model_file = args.model
output_file = args.output

import numpy as np
import gv
import amitgroup as ag
import glob
from skimage.transform import pyramid_reduce
from settings import load_settings

settings = load_settings(settings_file)
detector = gv.Detector.load(model_file)
descriptor = detector.descriptor

def create_bkg_generator(size, files):
    i = 0
    prng = np.random.RandomState(0)
    while True:
        im = gv.img.asgray(gv.img.load_image(files[i]))
        x, y = [prng.randint(0, im.shape[i]-size[i]-1) for i in xrange(2)]
        yield im[x:x+size[0],y:y+size[1]]
        i = (i + 1) % len(files)
         
# Iterate through the original CAD images. Superimposed them onto random background
neg_files = sorted(glob.glob(settings['detector']['neg_dir']))
cad_files = sorted(glob.glob(settings['detector']['train_dir']))

size = settings['detector']['image_size']

bkg_generator = create_bkg_generator(size, neg_files)

# First iterate through the background images to 
def limit(gen, N):
    for i in xrange(N):
        yield gen.next() 

pi = np.zeros(descriptor.num_parts)
pi_spread = np.zeros(descriptor.num_parts)
tot = 0
tot_spread = 0
cut = 4
radii = settings['detector']['spread_radii']
subsize = settings['detector']['subsample_size']

print 'Checking background model'

for bkg in limit(create_bkg_generator(size, neg_files), len(cad_files)):
    feats = descriptor.extract_features(bkg, settings=dict(spread_radii=(0, 0), subsample_size=(1, 1)))
    x = np.rollaxis(feats[cut:-cut,cut:-cut], 2).reshape((descriptor.num_parts, -1))
    tot += x.shape[1]
    pi += x.sum(axis=1)

    feats_spread = descriptor.extract_features(bkg, settings=dict(spread_radii=radii, subsample_size=(1, 1)))
    x_spread = np.rollaxis(feats_spread[cut:-cut,cut:-cut], 2).reshape((descriptor.num_parts, -1))
    tot_spread += x_spread.shape[1]
    pi_spread += x_spread.sum(axis=1)
    
unspread_bkg = pi / tot
spread_bkg = pi_spread / tot_spread

# Create model
print 'Creating model'

#a = 1 - unspread_bkg.sum()
#bkg_categorical = np.concatenate(([a], unspread_bkg))

#C = detector.kernel_basis * np.expand_dims(bkg_categorical, -1)
#kernels = C.sum(axis=-2) / detector.kernel_basis_samples.reshape((-1,) + (1,)*(C.ndim-2))

#kernels = np.clip(kernels, 1e-5, 1-1e-5)
kernels = detector.prepare_kernels(unspread_bkg)

comps = detector.mixture.mixture_components()

#theta = kernels.reshape((kernels.shape[0], -1))

llhs = [[] for i in xrange(detector.num_mixtures)] 

print 'Iterating CAD images'

for cad_i, cad_filename in enumerate(cad_files):
    cad = gv.img.load_image(cad_filename)
    cad = pyramid_reduce(cad, downscale=cad.shape[0]/size[0])
    mixcomp = comps[cad_i]

    alpha = cad[...,3]
    gray_cad = gv.img.asgray(cad)
    bkg_img = bkg_generator.next()
    
    # Notice, gray_cad is not multiplied by alpha, since most files use premultiplied alpha 
    composite = gray_cad + bkg_img * (1 - alpha)

    # Get features
    X_full = descriptor.extract_features(composite, settings=dict(spread_radii=radii))

    X = gv.sub.subsample(X_full, subsize)


    a = np.log(kernels[mixcomp]/(1-kernels[mixcomp]) * ((1-spread_bkg)/spread_bkg))

    # Check log-likelihood
    llh = np.sum(X * a)
    llhs[mixcomp].append(llh)
    
detector.fixed_train_mean = np.asarray([np.mean(llhs[k]) for k in xrange(detector.num_mixtures)]) 
detector.fixed_train_std = np.asarray([np.std(llhs[k]) for k in xrange(detector.num_mixtures)])
detector.kernel_templates = kernels
detector.kernel_basis = None
detector.fixed_bkg = unspread_bkg
detector.settings['bkg_type'] = 'from-file'
detector.settings['testing_type'] = 'fixed'
#detector.settings['train_unspread'] = False
detector.settings['kernels_ready'] = True

detector.save(output_file)
