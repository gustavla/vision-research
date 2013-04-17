from __future__ import division
import glob
import numpy as np
import amitgroup as ag
import gv
import os
import sys
from itertools import product
from superimpose_experiment import generate_random_patches


# TODO: Temporary
sh = (28, 88)


def composite(fg_img, bg_img, alpha):
    return fg_img * alpha + bg_img * (1 - alpha) 

def create_graymap(size, shade, prnd):
    graymap = np.empty(size)
    graymap.fill(shade)
            
    # Add gaussian noise, to stimulate different edge activity
    # It has to be well below the minimum contrast threshold though,
    # so that we don't stimulate edges outside the object.                  
    graymap = np.clip(graymap + prnd.randn(*size) * 0.02, 0, 1)

    return graymap
     
def weighted_choice_unit(x, randgen=np.random):
    xcum = np.cumsum(x) 
    r = randgen.rand()
    w = np.where(r < xcum)[0]
    if w.size == 0:
        return -1
    else:
        return w[0]
    return 

def get_probs(theta, f):
    if f == -1:
        return 0
    else:
        return theta[f]

def _process_file_star(args):
    return _process_file(*args)

def _process_file(settings, bkg_stack, bkg_stack_num, fn):
    ag.info("Processing file", fn)
    seed = np.abs(hash(fn)%123124)

    # We need the descriptor to generate and manipulate images
    descriptor = gv.load_descriptor(settings)

    counts = np.zeros((1, sh[0], sh[1], descriptor.num_parts, descriptor.num_parts), dtype=np.uint16)

    prnds = [np.random.RandomState(seed+i) for i in xrange(5)]

    # Which mixture component does this image belong to?
    # TODO: Temporary until multicomp
    mixcomp = 0#np.argmax(detector.affinities

    # Binarize support and Extract alpha
    #color_img, alpha = gv.img.load_image_binarized_alpha(fn)
    color_img = gv.img.load_image(fn)
    alpha = color_img[...,3]
    img = gv.img.asgray(color_img) 

    # Settings
    bsettings = settings['edges'].copy()
    radius = bsettings['radius']
    bsettings['radius'] = 0

    locations0 = xrange(sh[0])
    locations1 = xrange(sh[1])

    #locations0 = xrange(10, 11)
    #locations1 = xrange(10, 11)

    #padded_theta = descriptor.unspread_parts_padded

    #pad = 10
    pad = 5 
    X_pad_size = (9+pad*2,)*2



    img_pad = ag.util.zeropad(img, pad)

    alpha_pad = ag.util.zeropad(alpha, pad)

    # Iterate every duplicate

    dups = settings['detector'].get('duplicates', 1)

    bkgs = np.empty((descriptor.num_parts * dups,) + X_pad_size) 
    #cads = np.empty((descriptor.num_parts,) + X_pad_size)
    #alphas = np.empty((descriptor.num_parts,) + X_pad_size, dtype=np.bool)

    #ag.info("Iteration {0}/{1}".format(loop+1, num_duplicates)) 
    #ag.info("Iteration")
    for i, j in product(locations0, locations1):
        selection = [slice(i, i+X_pad_size[0]), 
                     slice(j, j+X_pad_size[1])]
        #X_pad = edges_pad[selection].copy()
        patch = img_pad[selection]
        alpha_patch = alpha_pad[selection]

        patch = np.expand_dims(patch, 0)
        alpha_patch = np.expand_dims(alpha_patch, 0)

        for f in xrange(descriptor.num_parts):
            num = bkg_stack_num[f]

            for d in xrange(dups):
                bkg_i = prnds[4].randint(num)
                bkgs[f*dups+d] = bkg_stack[f,bkg_i]
            
        
        img_with_bkgs = patch * alpha_patch + bkgs * (1 - alpha_patch)

        edges_pads = ag.features.bedges(img_with_bkgs, **bsettings)
        X_pad_spreads = ag.features.bspread(edges_pads, spread=bsettings['spread'], radius=radius)

        padding = pad - 2
        X_spreads = X_pad_spreads[:,padding:-padding:,padding:-padding]

        partprobs = ag.features.code_parts_many(X_spreads, descriptor._log_parts, descriptor._log_invparts, 
                                                descriptor.settings['threshold'], descriptor.settings['patch_frame'])

        parts = partprobs.argmax(axis=-1)

        for f in xrange(descriptor.num_parts):
            hist = np.bincount(parts[f*dups:(f+1)*dups].ravel(), minlength=descriptor.num_parts+1)
            counts[mixcomp,i,j,f] += hist[1:]

        #import pdb; pdb.set_trace()

        #for f in xrange(descriptor.num_parts):
        #    for d in xrange(dups):
        #        # Code parts 
        #        #parts = descriptor.extract_parts(X_spreads[f*dups+d].astype(np.uint8))
#
#                f_plus = parts[f*dups+d]
#                if f_plus > 0:
        #tau = self.settings.get('tau')
        #if self.settings.get('tau'):
        #parts = partprobs.argmax(axis=-1)

                # Accumulate and return
#                    counts[mixcomp,i,j,f,f_plus-1] += 1#parts[0,0]

        if 0:
            for f in xrange(descriptor.num_parts):

                # Pick only one background for this part and file
                num = bkg_stack_num[f]
        
                # Assumes num > 0

                bkg_i = prnds[4].randint(num)

                bkgmap = bkg_stack[f,bkg_i]

                # Composite
                img_with_bkg = composite(patch, bkgmap, alpha_patch)

                # Retrieve unspread edges (with a given background gray level) 
                edges_pad = ag.features.bedges(img_with_bkg, **bsettings)

                # Pad the edges
                #edges_pad = ag.util.zeropad(edges, (pad, pad, 0)) 

                # Do spreading
                X_pad_spread = ag.features.bspread(edges_pad, spread=bsettings['spread'], radius=radius)

                # De-pad
                padding = pad - 2
                X_spread = X_pad_spread[padding:-padding,padding:-padding]

                # Code parts 
                parts = descriptor.extract_parts(X_spread.astype(np.uint8))

                # Accumulate and return
                counts[mixcomp,i,j,f] += parts[0,0]

    return counts

def background_adjust_model(settings, bkg_stack, bkg_stack_num, seed=0):
    offset = settings['detector'].get('train_offset', 0)
    limit = settings['detector'].get('train_limit')
    assert limit is not None, "Must specify limit in the settings file"
    duplicates = settings['detector'].get('duplicates', 1)
    files = sorted(glob.glob(settings['detector']['train_dir']))[offset:limit]

    try:
        detector = gv.Detector.load(settings['detector']['file'])
    except KeyError:
        raise Exception("Need to train the model first")

    # Create accumulates for each mixture component
    # TODO: Temporary until multicomp
    #counts = np.zeros_like(detector.kernel_templates)

    #num_files = len(files)
    #num_duplicates = settings['detector'].get('duplicate', 1)

    # Create several random states, so it's easier to measure
    # the influence of certain features

    # Setup unspread bedges settings
    #X_pad_size = padded_theta.shape[1:3]

    #for fn in files:
        #counts += _process_file(settings, bkg_stack, bkg_stack_num, fn)

    if 1:
        from multiprocessing import Pool
        p = Pool(7)
        # Important to run imap, since otherwise we will accumulate too
        # much memory, since the count structure is quite big.
        imapf = p.imap_unordered
    else:
        from itertools import imap as imapf

    argses = [(settings, bkg_stack, bkg_stack_num, fn) for fn in files]

    # Iterate images
    all_counts = imapf(_process_file_star, argses)

    # Can dot his instead:
    counts = sum(all_counts)

    # Divide accmulate to get new distribution
    #counts /= num_files
    
    # Create a new model, with this distribution
    #new_detector = detector.copy() 

    #new_detector.kernel_templates = counts
    #new_detector.support = None
    #new_detector.use_alpha = False

    # Return model 
    #return new_detector
    return counts, limit * duplicates


if __name__ == '__main__':
    import argparse
    from settings import load_settings
   
    ag.set_verbose(True)
    
    parser = argparse.ArgumentParser(description="Convert model to integrate background model")
    parser.add_argument('settings', metavar='<settings file>', type=argparse.FileType('r'), help='Filename of settings file')
    parser.add_argument('bkgstack', metavar='<bkgstack file>', type=argparse.FileType('rb'), help='Background stack model file')
    parser.add_argument('output', metavar='<output file>', type=argparse.FileType('wb'), help='Model output file')
    
    args = parser.parse_args()
    settings_file = args.settings
    bkg_file = args.bkgstack
    output_file = args.output

    settings = load_settings(settings_file)
    bkg_data = np.load(bkg_file)
    bkg_stack = bkg_data['bkg_stack']
    bkg_stack_num = bkg_data['bkg_stack_num']

    counts, total = background_adjust_model(settings, bkg_stack, bkg_stack_num)
    
    descriptor = gv.load_descriptor(settings)

    # Create the model file 
    detector = gv.Detector(settings['detector']['num_mixtures'], descriptor, settings['detector'])
    detector.kernel_basis = counts
    detector.kernel_basis_samples = total
    detector.use_alpha = False
    # TODO: Temporary
    detector.orig_kernel_size = (40, 100)
    detector.save(output_file)

    

    #np.save(output_file, counts)
