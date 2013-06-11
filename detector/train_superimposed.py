from __future__ import division
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import glob
import numpy as np
import amitgroup as ag
import gv
import os
import sys
from itertools import product, cycle
from superimpose_experiment import generate_random_patches

def generate_random_patches(filenames, size, seed=0, per_image=1):
    randgen = np.random.RandomState(seed)
    failures = 0
    for fn in cycle(filenames):
        img = gv.img.resize_with_factor_new(gv.img.asgray(gv.img.load_image(fn)), randgen.uniform(0.5, 1.0))

        for l in xrange(per_image):
            # Random position
            x_to = img.shape[0]-size[0]-1
            y_to = img.shape[1]-size[1]-1

            if x_to >= 1 and y_to >= 1:
                x = randgen.randint(x_to) 
                y = randgen.randint(y_to)
                yield img[x:x+size[0], y:y+size[1]]
                failures = 0
            else:
                failures += 1

            # The images are too small, let's stop iterating
            if failures >= 30:
                return

def _create_kernel_for_mixcomp(mixcomp, settings, indices, files, neg_files):
    size = settings['detector']['image_size']

    gen = generate_random_patches(neg_files, size, seed=mixcomp)
    descriptor = gv.load_descriptor(settings)

    eps = settings['detector']['min_probability']
    radii = settings['detector']['spread_radii']
    psize = settings['detector']['subsample_size']
    duplicates = settings['detector'].get('duplicates', 1)
    cb = settings['detector'].get('crop_border')

    kern = None
    total = 0

    for index in indices: 
        ag.info("Processing image of index {0} and mixture component {1}".format(index, mixcomp))
        im = gv.img.resize(gv.img.load_image(files[index]), size)
        gray_im, alpha = gv.img.asgray(im), im[...,3] 

        for dup in xrange(duplicates):
            neg_im = gen.next()
            superimposed_im = neg_im * (1 - alpha) + gray_im 

            feats = descriptor.extract_features(superimposed_im, settings=dict(spread_radii=radii, crop_border=cb))
            feats = gv.sub.subsample(feats, psize)

            if kern is None:
                kern = feats.astype(np.uint32)
            else:
                kern += feats

            total += 1
    
    kern = kern.astype(np.float64) / total 
    kern = np.clip(kern, eps, 1-eps)

    #kernels.append(kern)
    return kern

def _create_kernel_for_mixcomp_star(args):
    return _create_kernel_for_mixcomp(*args)
        
def _calc_standardization_for_mixcomp(mixcomp, settings, kern, bkg, indices, files, neg_files):
    size = settings['detector']['image_size']

    # Use the same seed for all mixture components! That will make them easier to compare,
    # without having to sample to infinity.
    gen = generate_random_patches(neg_files, size, seed=0)
    descriptor = gv.load_descriptor(settings)

    eps = settings['detector']['min_probability']
    radii = settings['detector']['spread_radii']
    psize = settings['detector']['subsample_size']
    duplicates = settings['detector'].get('duplicates', 1) * 10 
    cb = settings['detector'].get('crop_border')

    total = 0

    llhs = []

    weights = np.log(kern / (1 - kern) * ((1 - bkg) / bkg))

    for index in indices: 
        ag.info("Standardizing image of index {0} and mixture component {1}".format(index, mixcomp))
        im = gv.img.resize(gv.img.load_image(files[index]), size)
        gray_im, alpha = gv.img.asgray(im), im[...,3] 

        for dup in xrange(duplicates):
            neg_im = gen.next()
            #superimposed_im = neg_im * (1 - alpha) + gray_im 
            superimposed_im = neg_im

            feats = descriptor.extract_features(superimposed_im, settings=dict(spread_radii=radii, crop_border=cb))
            feats = gv.sub.subsample(feats, psize)
        
            llh = (weights * feats).sum()
            llhs.append(llh)

    np.save('llhs-{0}.npy'.format(mixcomp), llhs)

    return np.mean(llhs), np.std(llhs)

def _calc_standardization_for_mixcomp_star(args):
    return _calc_standardization_for_mixcomp(*args)

def superimposed_model(settings, threading=True):
    offset = settings['detector'].get('train_offset', 0)
    limit = settings['detector'].get('train_limit')
    num_mixtures = settings['detector']['num_mixtures']
    assert limit is not None, "Must specify limit in the settings file"
    files = sorted(glob.glob(settings['detector']['train_dir']))[offset:offset+limit]
    neg_files = sorted(glob.glob(settings['detector']['neg_dir']))

    # Train a mixture model to get a clustering of the angles of the object
    descriptor = gv.load_descriptor(settings)
    detector = gv.Detector(num_mixtures, descriptor, settings['detector'])
    detector.train_from_images(files)

    comps = detector.mixture.mixture_components()
    each_mix_N = np.bincount(comps, minlength=num_mixtures)

    for fn in glob.glob('toutputs/*.png'):
        os.remove(fn)

    from shutil import copyfile
    for mixcomp in xrange(detector.num_mixtures):
        indices = np.where(comps == mixcomp)[0]
        for i in indices:
            copyfile(files[i], 'toutputs/mixcomp-{0}-index-{1}.png'.format(mixcomp, i))

    support = detector.support 

    kernels = []

    #print "TODO, quitting"
    #return detector


    #for mixcomp in xrange(num_mixtures):
    
    if threading:
        from multiprocessing import Pool
        p = Pool(7)
        # Order is important, so we can't use imap_unordered
        imapf = p.imap
    else:
        from itertools import imap as imapf
    
    argses = [(i, settings, list(np.where(comps == i)[0]), files, neg_files) for i in xrange(detector.num_mixtures)] 
    #argses = [(i,) for i in xrange(detector.num_mixtures)] 
    kernels = []
    for kern in imapf(_create_kernel_for_mixcomp_star, argses):
        kernels.append(kern)

    detector.kernel_templates = kernels
    detector.settings['kernel_ready'] = True
    detector.use_alpha = False
    detector.support = support

    # Determine the background
    ag.info("Determining background")
    #spread_bkg = np.mean([kern[:2].reshape((-1, kern.shape[-1])).mean(axis=0) for kern in kernels], axis=0)
    #spread_bkg = np.mean([kern.reshape((-1, kern.shape[-1])).mean(axis=0) for kern in kernels], axis=0)
    spread_bkg = kernels[0][1].mean(axis=0)

    print 'spread_bkg shape:', spread_bkg.shape
    detector.fixed_bkg = None # Not needed, since kernel_ready is True
    detector.fixed_spread_bkg = spread_bkg
    detector.settings['bkg_type'] = 'from-file'

    # Determine the standardization values
    ag.info("Determining standardization values")

    detector.fixed_train_mean = np.zeros(detector.num_mixtures)
    detector.fixed_train_std = np.zeros(detector.num_mixtures)
    
    argses = [(i, settings, kernels[i], spread_bkg, list(np.where(comps == i)[0]), files, neg_files) for i in xrange(detector.num_mixtures)]
    for i, (mean, std) in enumerate(imapf(_calc_standardization_for_mixcomp_star, argses)):
        detector.fixed_train_mean[i] = mean
        detector.fixed_train_std[i] = std

    detector.settings['testing_type'] = 'fixed'

    return detector 

    if 0:
        if threading:
            from multiprocessing import Pool
            p = Pool(7)
            # Important to run imap, since otherwise we will accumulate too
            # much memory, since the count structure is quite big.
            imapf = p.imap_unordered
        else:
            from itertools import imap as imapf

        argses = [(settings, files[i], comps[i]) for i in xrange(len(files))] 

        all_counts = imapf(_process_file_star, argses)
    

if __name__ == '__main__':
    import argparse
    from settings import load_settings
   
    ag.set_verbose(True)
    
    parser = argparse.ArgumentParser(description="Convert model to integrate background model")
    parser.add_argument('settings', metavar='<settings file>', type=argparse.FileType('r'), help='Filename of settings file')
    parser.add_argument('output', metavar='<output file>', type=argparse.FileType('wb'), help='Model output file')
    parser.add_argument('--no-threading', action='store_true', default=False, help='Turn off threading')

    args = parser.parse_args()
    settings_file = args.settings
    output_file = args.output
    threading = not args.no_threading

    settings = load_settings(settings_file)

    detector = superimposed_model(settings, threading=threading)

    #detector = gv.Detector(settings['detector']['num_mixtures'], descriptor, settings['detector'])
    #detector.kernel_templates = 

    detector.save(output_file)
