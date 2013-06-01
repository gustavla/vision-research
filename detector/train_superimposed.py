from __future__ import division
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pylab as plt
import glob
import numpy as np
import amitgroup as ag
import gv
import os
import sys
import ipdb
from itertools import product, cycle
from superimpose_experiment import generate_random_patches

def generate_random_patches(filenames, size, seed=0, per_image=1):
    randgen = np.random.RandomState(seed)
    failures = 0
    for fn in cycle(filenames):
        img = gv.img.asgray(gv.img.load_image(fn))
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

def handle_mixcomp(mixcomp, settings, indices, files, neg_files):
    size = settings['detector']['image_size']

    gen = generate_random_patches(neg_files, size, seed=mixcomp)
    descriptor = gv.load_descriptor(settings)

    eps = settings['detector']['min_probability']
    radii = settings['detector']['spread_radii']
    psize = settings['detector']['subsample_size']
    duplicates = settings['detector'].get('duplicates', 1)

    kern = None
    total = 0

    for index in indices: 
        ag.info("Processing image of index {0} and mixture component {1}".format(index, mixcomp))
        im = gv.img.load_image(files[index])
        gray_im, alpha = gv.img.asgray(im), im[...,3] 

        for dup in xrange(duplicates):
            neg_im = gen.next()
            superimposed_im = neg_im * (1 - alpha) + gray_im 

            feats = descriptor.extract_features(superimposed_im, settings=dict(spread_radii=radii))
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

def handle_mixcomp_star(args):
    return handle_mixcomp(*args)
        

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
    print comps
    each_mix_N = np.bincount(comps, minlength=num_mixtures)

    support = detector.support 

    kernels = []


    #for mixcomp in xrange(num_mixtures):
    
    if threading:
        from multiprocessing import Pool
        p = Pool(7)
        # Important to run imap, since otherwise we will accumulate too
        # much memory, since the count structure is quite big.
        imapf = p.imap#_unordered
    else:
        from itertools import imap as imapf
    
    argses = [(i, settings, list(np.where(comps == i)[0]), files, neg_files) for i in xrange(detector.num_mixtures)] 
    #argses = [(i,) for i in xrange(detector.num_mixtures)] 
    kernels = list(imapf(handle_mixcomp_star, argses))

    detector.kernel_templates = kernels
    detector.settings['kernel_ready'] = True
    detector.use_alpha = False
    detector.support = support

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
