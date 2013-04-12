from __future__ import division
import glob
import numpy as np
import amitgroup as ag
import gv
import os
from itertools import product
from superimpose_experiment import generate_random_patches

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

def background_adjust_model(settings, bkg, seed=0):
    offset = settings['detector'].get('train_offset', 0)
    limit = settings['detector'].get('train_limit')
    files = sorted(glob.glob(settings['detector']['train_dir']))[offset:limit]# * settings['detector'].get('duplicate', 1)

    try:
        detector = gv.Detector.load(settings['detector']['file'])
    except KeyError:
        raise Exception("Need to train the model first")

    # We need the descriptor to generate and manipulate images
    descriptor = gv.load_descriptor(settings)

    sh = (28, 88)

    # Create accumulates for each mixture component
    # TODO: Temporary until multicomp
    #counts = np.zeros_like(detector.kernel_templates)
    counts = np.zeros((1, sh[0], sh[1], descriptor.num_parts))

    num_files = len(files)
    num_duplicates = settings['detector'].get('duplicate', 1)

    # Create several random states, so it's easier to measure
    # the influence of certain features
    prnds = [np.random.RandomState(seed+i) for i in xrange(10)]

    # Setup unspread bedges settings
    bsettings = settings['edges'].copy()
    radius = bsettings['radius']
    bsettings['radius'] = 0

    locations0 = xrange(sh[0])
    locations1 = xrange(sh[1])

    padded_theta = descriptor.unspread_parts_padded

    #pad = 10
    pad = 5 
    X_pad_size = (9+pad*2,)*2
    #X_pad_size = padded_theta.shape[1:3]

    neg_filenames= sorted(glob.glob(os.path.join(os.environ['UIUC_DIR'], 'TrainImages', 'neg-*.pgm')))

    for seed, fn in enumerate(files):
        ag.info("Processing file", fn)

        # Which mixture component does this image belong to?
        # TODO: Temporary until multicomp
        mixcomp = 0#np.argmax(detector.affinities

        # Binarize support and Extract alpha
        color_img, alpha = gv.img.load_image_binarized_alpha(fn)
        img = gv.img.asgray(color_img) 

        img_pad = ag.util.zeropad(img, pad)

        alpha_pad = ag.util.zeropad(alpha, pad)
        inv_alpha_pad_expanded = np.expand_dims(~alpha_pad, -1)

        gen = generate_random_patches(neg_filenames, X_pad_size, seed)

        # Iterate every duplicate

        #ag.info("Iteration {0}/{1}".format(loop+1, num_duplicates)) 
        #ag.info("Iteration")
        for i, j in product(locations0, locations1):
            for loop in xrange(num_duplicates):
                selection = [slice(i, i+X_pad_size[0]), 
                             slice(j, j+X_pad_size[1])]
                #X_pad = edges_pad[selection].copy()
                patch = img_pad[selection]
                alpha_patch = alpha_pad[selection]

                bkgmap = gen.next()

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
                counts[mixcomp,i,j] += parts[0,0]

    """
    if 0:
        from multiprocessing import Pool
        p = Pool(7)
        mapf = p.map
    else:
        mapf = map
    def _process_file(fn): 
        return _process_file_full(fn, sh, descriptor, detector)

    # Iterate images
    all_counts = mapf(_process_file, files)

    for counti in all_counts:
        counts += counti
    """

    # Divide accmulate to get new distribution
    counts /= num_files * num_duplicates
    
    # Create a new model, with this distribution
    new_detector = detector.copy() 

    new_detector.kernel_templates = counts
    new_detector.support = None
    new_detector.use_alpha = False

    # Return model 
    return new_detector


if __name__ == '__main__':
    import argparse
    from settings import load_settings
   
    ag.set_verbose(True)
    
    parser = argparse.ArgumentParser(description="Convert model to integrate background model")
    parser.add_argument('settings', metavar='<settings file>', type=argparse.FileType('r'), help='Filename of settings file')
    parser.add_argument('bkg', metavar='<background file>', type=argparse.FileType('rb'), help='Background model file')
    parser.add_argument('output', metavar='<output file>', type=argparse.FileType('wb'), help='Model output file')
    
    args = parser.parse_args()
    settings_file = args.settings
    bkg_file = args.bkg
    output_file = args.output

    settings = load_settings(settings_file)
    bkg = np.load(bkg_file)

    model = background_adjust_model(settings, bkg)

    model.save(output_file)
