from __future__ import division, print_function, absolute_import
import glob
import numpy as np
import amitgroup as ag
import gv
import os
import sys
import itertools as itr
from collections import namedtuple
from copy import copy
from superimpose_experiment import generate_random_patches
from gv.keypoints import get_key_points
from scipy.special import logit, expit


#KMEANS = False 
#LOGRATIO = True 
SVM_INDICES = False#True
INDICES = True 
#LOGRATIO = True 
LOGRATIO = False
LLH_NEG = True

#Patch = namedtuple('Patch', ['filename', 'selection'])

#def load_patch_image(patch):
#    img = gv.img.asgray(gv.img.load_image(patch.filename))
#    return img[patch.selection]

def generate_random_patches(filenames, size, seed=0, per_image=1):
    filenames = copy(filenames)
    randgen = np.random.RandomState(seed)
    randgen.shuffle(filenames)
    failures = 0
    for fn in itr.cycle(filenames):
        #img = gv.img.resize_with_factor_new(gv.img.asgray(gv.img.load_image(fn)), randgen.uniform(0.5, 1.0))
        img = gv.img.asgray(gv.img.load_image(fn))

        for l in xrange(per_image):
            # Random position
            x_to = img.shape[0]-size[0]+1
            y_to = img.shape[1]-size[1]+1

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

def generate_feature_patches_dense(filenames, size, extract_func, subsample_size, seed=0):
    filenames = copy(filenames)
    randgen = np.random.RandomState(seed)
    randgen.shuffle(filenames)
    failures = 0
    real_size = extract_func(np.zeros(size)).shape[:2]
    for fn in itr.cycle(filenames):
        #img = gv.img.resize_with_factor_new(gv.img.asgray(gv.img.load_image(fn)), randgen.uniform(0.5, 1.0))
        img = gv.img.asgray(gv.img.load_image(fn))
        feats = extract_func(img)

        dx, dy = [feats.shape[i]-real_size[i] for i in xrange(2)]
        if min(dx, dy) == 0:
            continue

        for x, y in itr.product(xrange(dx), xrange(dy)):
            xx = x*subsample_size[0]
            yy = y*subsample_size[1]
            try:
                assert xx+size[0] < img.shape[0]
                assert yy+size[1] < img.shape[1]
            except:
                import pdb; pdb.set_trace()
            yield img[xx:xx+size[0],yy:yy+size[1]], feats[x:x+real_size[0],y:y+real_size[1]], x, y, img

def _create_kernel_for_mixcomp(mixcomp, settings, bb, indices, files, neg_files):
    im_size = settings['detector']['image_size']
    size = gv.bb.size(bb)
    orig_size = size
    
    gen = generate_random_patches(neg_files, size, seed=0)
    
    descriptor = gv.load_descriptor(settings)

    radii = settings['detector']['spread_radii']
    psize = settings['detector']['subsample_size']
    duplicates = settings['detector'].get('duplicates', 1)
    cb = settings['detector'].get('crop_border')

    totals = 0
    bkg = None
    kern = None
    alpha_cum = None

    setts = dict(spread_radii=radii, subsample_size=psize, crop_border=cb)
    counts = 0 

    all_b = []
    all_X = []
    all_s = []

    for index in indices: 
        ag.info("Processing image of index {0} and mixture component {1}".format(index, mixcomp))
        gray_im, alpha = _load_cad_image(files[index], im_size, bb)

        bin_alpha = (alpha > 0.05).astype(np.uint32)

        if alpha_cum is None:
            alpha_cum = bin_alpha
        else:
            alpha_cum += bin_alpha 

        for dup in xrange(duplicates):
            neg_im = gen.next()
            neg_feats = descriptor.extract_features(neg_im, settings=setts)
            superimposed_im = neg_im * (1 - alpha) + gray_im * alpha

            feats = descriptor.extract_features(superimposed_im, settings=setts)

            counts += 1

            #bkg_feats = gv.sub.subsample(bkg_feats, psize)
        
            if bkg is None:
                bkg = neg_feats.astype(np.uint32)
            else:
                bkg += neg_feats

            #feats = gv.sub.subsample(feats, psize)

            if kern is None:
                kern = feats.astype(np.uint32)
            else:
                kern += feats

            # NEW TODO: This throws out low-activity negatives
            #if abs(neg_feats.mean() - 0.2) < 0.05:
            #if neg_feats.mean() < 0.05:
            if True:
                all_b.append(neg_feats)
                all_X.append(feats)
                all_s.append(bin_alpha)

                totals += 1

    print('COUNTS', counts)

    np.seterr(divide='raise')

    try:
        kern = kern.astype(np.float64) / totals
        bkg = bkg.astype(np.float64) / totals
    except:
        import pdb; pdb.set_trace()
    
    #kern = kern.astype(np.float64) / total 
    #kern = np.clip(kern, eps, 1-eps)

    #bkg = bkg.astype(np.float64) / total

    support = alpha_cum.astype(np.float64) / len(indices)

    return kern, bkg, orig_size, support 



def _load_cad_image(fn, im_size, bb):
    im = gv.img.load_image(fn)
    im = gv.img.resize(im, im_size)
    im = gv.img.crop_to_bounding_box(im, bb)
    if im.ndim == 3:
        if im.shape[2] == 4:
            gray_im, alpha = gv.img.asgray(im), im[...,3] 
        else:
            gray_im = gv.img.asgray(im)
            alpha = np.zeros(gray_im.shape)
    else:
        gray_im = im
        alpha = np.zeros(gray_im.shape)

    return gray_im, alpha
        

def _calc_standardization_for_mixcomp(mixcomp, settings, eps, bb, kern, bkg, indices_UNUSED, files_UNUSED, neg_files_UNUSED, weight_indices, duplicates_mult=1):
    clipped_kern = np.clip(kern, eps, 1 - eps)
    clipped_bkg = np.clip(bkg, eps, 1 - eps)
    weights = gv.BernoulliDetector.build_weights(clipped_kern, clipped_bkg)

    standardization_info = []

    llh_mean = 0.0
    llh_var = 0.0
    if weight_indices is not None:
        for index in weight_indices:
            part = index[-1]
            mvalue = clipped_bkg[...,part].mean()

            llh_mean += mvalue * weights[tuple(index)]
            llh_var += mvalue * (1 - mvalue) * weights[tuple(index)]**2
    else:
        llh_mean = (clipped_bkg * weights).sum()
        llh_var = (clipped_bkg * (1 - clipped_bkg) * weights**2).sum()

    info = {}
    info['mean'] = llh_mean 
    info['std'] = np.sqrt(llh_var)

    return info 



def _get_positives(mixcomp, settings, indices, files):
    im_size = settings['detector']['image_size']

    # Use the same seed for all mixture components! That will make them easier to compare,
    # without having to sample to infinity.


    # HERE: Make it possible to input data directly!
    descriptor = gv.load_descriptor(settings)

    radii = settings['detector']['spread_radii']
    psize = settings['detector']['subsample_size']
    cb = settings['detector'].get('crop_border')

    #obj_counts = None
    #count = 0
    all_feats = []

    for index in indices: 
        ag.info("Fetching positives from image of index {0} and mixture component {1}".format(index, mixcomp))
        gray_im = gv.img.asgray(gv.img.load_image(files[index]))
        gray_im = gv.img.resize(gray_im, im_size)

        feats = descriptor.extract_features(gray_im, settings=dict(spread_radii=radii, subsample_size=psize, crop_border=cb))
        all_feats.append(feats)
        if 0:
            if obj_counts is None:
                obj_counts = feats.astype(int)
            else:
                obj_counts += feats

            count += 1 

    #assert count > 0, "Did not find any object images!"

    #return obj_counts.astype(np.float64) / count
    return np.asarray(all_feats)

def __process_bkg(fn, descriptor, sett, factor):
    im = gv.img.asgray(gv.img.load_image(fn))
    im = gv.img.resize_with_factor_new(im, factor)


    ag.info("Processing image for background model:", fn)
    feats = descriptor.extract_features(im, settings=sett)

    #count += np.prod(feats.shape[:2])
    #bkg_counts += np.apply_over_axes(np.sum, feats, [0, 1]).ravel()
    return np.apply_over_axes(np.sum, feats, [0, 1]).ravel(), np.prod(feats.shape[:2])

def _get_background_model(settings, neg_files):
    descriptor = gv.load_descriptor(settings)
    neg_count = settings['detector'].get('train_neg_limit', 50)

    rs = np.random.RandomState(0)

    radii = settings['detector']['spread_radii']
    psize = settings['detector']['subsample_size']
    cb = settings['detector'].get('crop_border')

    bkg_counts = np.zeros(descriptor.num_features, dtype=int)
    count = 0

    sett = dict(spread_radii=radii, subsample_size=psize, crop_border=cb)
    factors = rs.uniform(0.2, 1.0, size=neg_count)
    argses = [(neg_files[i], descriptor, sett, factors[i]) for i in xrange(neg_count)]

    for feats, c in gv.parallel.starmap_unordered(__process_bkg, argses):
    #for fn in itr.islice(neg_files, neg_count):
        if 0:
            im = gv.img.asgray(gv.img.load_image(fn))
            # Randomly resize
            factor = rs.uniform(0.2, 1.0)
            im = gv.img.resize_with_factor_new(im, factor)

            print(im.shape)

            feats = descriptor.extract_features(im, settings=sett)

            count += np.prod(feats.shape[:2])
            bkg_counts += np.apply_over_axes(np.sum, feats, [0, 1]).ravel()

        count += c 
        bkg_counts += feats 

    assert count > 0, "Did not find any background images!"
    
    bkg = bkg_counts.astype(np.float64) / count
    return bkg
    
def _process_file_kernel_basis(seed, mixcomp, settings, bb, filename, bkg_stack, bkg_stack_num):
    ag.info("Processing file ", filename)
    im_size = settings['detector']['image_size']
    size = gv.bb.size(bb)

    # Use the same seed for all mixture components! That will make them easier to compare,
    # without having to sample to infinity.


    # HERE: Make it possible to input data directly!
    descriptor = gv.load_descriptor(settings)

    part_size = descriptor.settings['part_size']
    radii = settings['detector']['spread_radii']
    psize = settings['detector']['subsample_size']
    cb = settings['detector'].get('crop_border')

    #sh = (size[0] // psize[0], size[1] // psize[1])
    sh = gv.sub.subsample_size_new((size[0]-4, size[1]-4), psize)

    all_pos_feats = []

    F = descriptor.num_features

    # No coding is also included
    counts = np.zeros(sh + (F + 1, F), dtype=np.int64)
    empty_counts = np.zeros((F + 1, F), dtype=np.int64)
    totals = 0

    sett = dict(spread_radii=radii, subsample_size=psize, crop_border=cb)


    alpha_maps = []

    gray_im, alpha = _load_cad_image(filename, im_size, bb)

    pad = (radii[0] + 2, radii[1] + 2)

    padded_gray_im = ag.util.zeropad(gray_im, pad)
    padded_alpha = ag.util.zeropad(alpha, pad)

    dups = 5
    X_pad_size = (part_size[0] + pad[0] * 2, part_size[1] + pad[1] * 2)

    bkgs = np.empty(((F + 1) * dups,) + X_pad_size) 

    rs = np.random.RandomState(seed)

    for f in xrange(F + 1):
    #for f in xrange(1):
        num = bkg_stack_num[f]

        for d in xrange(dups):
            bkg_i = rs.randint(num)
            bkgs[f*dups+d] = bkg_stack[f,bkg_i]

    # Do it with no superimposed image, to see what happens to pure background
    img_with_bkgs = bkgs
    #ex = descriptor.extract_features(img_with_bkgs[0], settings=sett)
    parts = np.asarray([descriptor.extract_features(im, settings=sett)[0,0] for im in img_with_bkgs])

    for f in xrange(F + 1):
        hist = parts[f*dups:(f+1)*dups].sum(0)
        empty_counts[f] += hist

    if 1:
        for i, j in itr.product(xrange(sh[0]), xrange(sh[1])):
            selection = [slice(i * psize[0], i * psize[0] + X_pad_size[0]), slice(j * psize[1], j * psize[1] + X_pad_size[1])]

            patch = padded_gray_im[selection]
            alpha_patch = padded_alpha[selection]

            patch = patch[np.newaxis]
            alpha_patch = alpha_patch[np.newaxis]

            img_with_bkgs = patch * alpha_patch + bkgs * (1 - alpha_patch)
            
            #ex = descriptor.extract_features(img_with_bkgs[0], settings=sett)
            parts = np.asarray([descriptor.extract_features(im, settings=sett)[0,0] for im in img_with_bkgs])

            #counts[i,j] += parts
            for f in xrange(F + 1):
            #for f in xrange(1):
                #hist = np.bincount(parts[f*dups:(f+1)*dups].ravel(), minlength=F + 1)
                hist = parts[f*dups:(f+1)*dups].sum(0)
                counts[i,j,f] += hist

    totals += dups 

    #support = alpha_maps.mean(axis=0)

    return counts, empty_counts, totals


def __process_one(args):
    index, mixcomp, files, im_size, bb, duplicates, neg_files, descriptor, sett = args
    size = gv.bb.size(bb)
    psize = sett['subsample_size']

    ADAPTIVE = True 
    if ADAPTIVE:
        # Do a pre-run, investigating the object model



        gen = generate_feature_patches_dense(neg_files, size, lambda im: descriptor.extract_features(im, settings=sett), psize, seed=index)
        dd = gv.Detector.load('uiuc-np3b.npy')

        support = dd.extra['sturf'][0]['support']
        pos = dd.extra['sturf'][0]['pos'].astype(bool)
        neg = dd.extra['sturf'][0]['neg'].astype(bool)
        S = support[...,np.newaxis]
        appeared = pos & ~neg
        A = appeared.mean(0) / (0.00001+((1-neg).mean(0)))
        obj = ((np.apply_over_axes(np.mean, (A*S), [0, 1])) / S.mean()).ravel()

        #obj = dd.extra['sturf'][0]['pavg']
        obj_clipped = gv.bclip(obj, 0.001)
        obj_std = np.sqrt(obj_clipped * (1 - obj_clipped))
        if 0:
            kern = dd.kernel_templates[mixcomp]
            obj = np.apply_over_axes(np.mean, kern, [0, 1]).squeeze()
            obj_clipped = np.clip(obj, 0.01, 1-0.01)
            obj_std = np.sqrt(obj_clipped * (1 - obj_clipped))
            #obj_std = np.ones(obj.size)
        from scipy.stats import norm
        running_avg = None
        C = 0
    else:
        gen = generate_random_patches(neg_files, size, seed=index)

    ag.info("Fetching positives from image of index {0} and mixture component {1}".format(index, mixcomp))
    gray_im, alpha = _load_cad_image(files[index], im_size, bb)

    all_pos_feats = []
    all_neg_feats = []
    for dup in xrange(duplicates):
        neg_feats = None

        if ADAPTIVE:
            best = (np.inf,)
            for i in xrange(2500):
                neg_im, neg_feats, x, y, im = gen.next()

                avg = np.apply_over_axes(np.mean, neg_feats, [0, 1]).squeeze()

                if C == 0:
                    mean_avg = avg
                else:
                    mean_avg = running_avg * (C - 1) / C + avg * 1 / C

                #dist = np.sum((avg - obj)**2)
                dist = -norm.logpdf((mean_avg - obj) / obj_std).sum()
                #if neg_feats.mean() > 0.02:
                    #break


                if dist < best[0]:
                    best = (dist, neg_im, neg_feats, mean_avg)

            running_avg = best[3] 

            C += 1

            print('dist', best[0])
            neg_im, neg_feats = best[1:3]
        else:
            neg_im = gen.next()
            neg_feats = descriptor.extract_features(neg_im, settings=sett)
            
        # Check which component this one is
        superimposed_im = neg_im * (1 - alpha) + gray_im * alpha

        pos_feats = descriptor.extract_features(superimposed_im, settings=sett)

        all_pos_feats.append(pos_feats)
        all_neg_feats.append(neg_feats)
    return all_pos_feats, all_neg_feats, alpha

def _get_pos_and_neg(mixcomp, settings, bb, indices, files, neg_files, duplicates_mult=1):
    im_size = settings['detector']['image_size']
    size = gv.bb.size(bb)

    # Use the same seed for all mixture components! That will make them easier to compare,
    # without having to sample to infinity.


    # HERE: Make it possible to input data directly!
    descriptor = gv.load_descriptor(settings)

    all_pos_feats = []
    all_neg_feats = []

    radii = settings['detector']['spread_radii']
    psize = settings['detector']['subsample_size']
    duplicates = settings['detector'].get('duplicates', 1) * duplicates_mult
    cb = settings['detector'].get('crop_border')

    sett = dict(spread_radii=radii, subsample_size=psize, crop_border=cb)


    alpha_maps = []

    # TODO {{{
    if 0:
        pass
    else:
        pass
    # }}}

    args = [(index, mixcomp, files, im_size, bb, duplicates, neg_files, descriptor, sett) for index in indices]
#index, mixcomp, files, im_size, bb, duplicates):
    for pos_feats, neg_feats, alpha in gv.parallel.imap_unordered(__process_one, args):
        all_pos_feats.extend(pos_feats)
        all_neg_feats.extend(neg_feats)
        alpha_maps.append(alpha)

    all_neg_feats = np.asarray(all_neg_feats)
    all_pos_feats = np.asarray(all_pos_feats)
    alpha_maps = np.asarray(alpha_maps)
    #support = alpha_maps.mean(axis=0)

    return all_neg_feats, all_pos_feats, alpha_maps 


def get_strong_fps(detector, i, fileobj):
    topsy = [[] for k in xrange(detector.num_mixtures)]
    #for i, fileobj in enumerate(itr.islice(gen, COUNT)):
    ag.info('{0} Farming {1}'.format(i, fileobj.img_id))
    img = gv.img.load_image(fileobj.path)
    grayscale_img = gv.img.asgray(img)

    for m in xrange(detector.num_mixtures):
        bbobjs = detector.detect_coarse(grayscale_img, fileobj=fileobj, mixcomps=[m], use_padding=False, use_scale_prior=False, cascade=True, more_detections=True)
        # Add in img_id
        for bbobj in bbobjs:
            bbobj.img_id = fileobj.img_id

            #array.append(bbobj.X)

            if bbobj.confidence > detector.extra['cascade_threshold']:
                #if len(topsy[m]) < TOP_N:
                #    heapq.heappush(topsy[m], bbobj)
                #else:
                #    heapq.heappushpop(topsy[m], bbobj)
                topsy[m].append(bbobj)

    return topsy


def superimposed_model(settings, threading=True):
    offset = settings['detector'].get('train_offset', 0)
    limit = settings['detector'].get('train_limit')
    num_mixtures = settings['detector']['num_mixtures']
    assert limit is not None, "Must specify limit in the settings file"
    files = sorted(glob.glob(settings['detector']['train_dir']))[offset:offset+limit]
    neg_files = sorted(glob.glob(settings['detector']['neg_dir']))
    neg_files2 = sorted(glob.glob(settings['detector']['neg_dir2']))

    # Train a mixture model to get a clustering of the angles of the object
    descriptor = gv.load_descriptor(settings)
    detector = gv.BernoulliDetector(num_mixtures, descriptor, settings['detector'])

    print("Checkpoint 1")

    testing_type = detector.settings.get('testing_type')
    bkg_type = detector.settings.get('bkg_type')


    # Extract clusters (manual or through EM)
    ##############################################################################

    if detector.num_mixtures == 1 or detector.settings.get('manual_clusters', False):

        if detector.num_mixtures == 1:
            comps = np.zeros(len(files), dtype=int)
        else:
            comps = np.zeros(len(files), dtype=np.int64)
            for i, f in enumerate(files):
                try:
                    v = int(os.path.basename(f).split('-')[0])
                    comps[i] = v
                except:
                    print("Name of training file ({}) not compatible with manual clustering".format(f), file=sys.stderr)
                    sys.exit(1)

        alpha_maps = []
        for i, grayscale_img, img, alpha in detector.load_img(files):
            alpha_maps.append(alpha)
        alpha_maps = np.asarray(alpha_maps)

        detector.determine_optimal_bounding_boxes(comps, alpha_maps)

    else:
        detector.settings['bkg_type'] = None
        detector.settings['testing_type'] = None

        detector.train_from_images(files)

        detector.settings['bkg_type'] = bkg_type
        detector.settings['testing_type'] = testing_type

        print("Checkpoint 2")

        comps = detector.mixture.mixture_components()
    each_mix_N = np.bincount(comps, minlength=num_mixtures)

    ##############################################################################

    print("Checkpoint 3")

    print("Checkpoint 4")

    support = detector.support 

    kernels = []

    #print("TODO, quitting")
    #return detector

    # Determine bounding boxes
    ##############################################################################

    psize = settings['detector']['subsample_size']

    def get_full_size_bb(k):
        bb = detector.bounding_box_for_mix_comp(k)
        return tuple(bb[i] * psize[i%2] for i in xrange(4))

    def iround(x):
        return int(round(x))

    def make_bb(bb, max_bb):
        # First, make it integral
        bb = (iround(bb[0]), iround(bb[1]), iround(bb[2]), iround(bb[3]))
        bb = gv.bb.inflate(bb, detector.settings.get('inflate_feature_frame', 4))
        bb = gv.bb.intersection(bb, max_bb)
        return bb

    print("Checkpoint 5")

    max_bb = (0, 0) + detector.settings['image_size']

    if 'bbs' in detector.extra:
        bbs = [make_bb(detector.extra['bbs'][k], max_bb) for k in xrange(detector.num_mixtures)]
    else: 
        bbs = [make_bb(get_full_size_bb(k), max_bb) for k in xrange(detector.num_mixtures)]

    print("Checkpoint 6")

    print("Checkpoint 7")

    kernels = []
    bkgs = []
    orig_sizes = []
    new_support = []
    im_size = settings['detector']['image_size']

    print("Checkpoint 8")
    all_negs = []

    print("Checkpoint 9")

    # Retrieve features and support 
    ##############################################################################

    ag.info('Fetching positives again...')
    all_pos_feats = []
    all_neg_feats = []
    alphas = []
    all_alphas = []
    all_binarized_alphas = []


    if settings['detector'].get('superimpose'):
        argses = [(m, settings, bbs[m], list(np.where(comps == m)[0]), files, neg_files, settings['detector'].get('stand_multiples', 1)) for m in range(detector.num_mixtures)]        
        for neg_feats, pos_feats, alpha_maps in itr.starmap(_get_pos_and_neg, argses):
            alpha = alpha_maps.mean(0)
            all_alphas.append(alpha_maps)
            all_binarized_alphas.append(alpha_maps > 0.05)

            alphas.append(alpha)
            all_neg_feats.append(neg_feats)
            all_pos_feats.append(pos_feats)

        ag.info('Done.')

        # Setup some places to store things
        if 'weights' not in detector.extra:
            detector.extra['weights'] = [None] * detector.num_mixtures
        if 'sturf' not in detector.extra:
            detector.extra['sturf'] = [{} for _ in xrange(detector.num_mixtures)]

        for m in xrange(detector.num_mixtures):
            detector.extra['sturf'].append(dict())

            obj = all_pos_feats[m].mean(axis=0)
            bkg = all_neg_feats[m].mean(axis=0)
            size = gv.bb.size(bbs[m])

            kernels.append(obj)
            bkgs.append(bkg)
            orig_sizes.append(size)
            new_support.append(alphas[m])

        if 0:
            for m in xrange(detector.num_mixtures):
                obj = all_pos_feats[m].mean(axis=0)
                bkg = all_neg_feats[m].mean(axis=0)
                size = gv.bb.size(bbs[m])

                eps = 0.025
                obj = np.clip(obj, eps, 1 - eps)
                avg = np.clip(avg, eps, 1 - eps)
                #lmb = obj / avg
                #w = np.clip(np.log(obj / avg), -1, 1)
                w = np.log(obj / (1 - obj) * ((1 - avg) / avg))
                #w = np.log(

                #w_avg = np.apply_over_axes(np.sum, w * support[...,np.newaxis], [0, 1]) / support.sum()

                #w -= w_avg * support[...,np.newaxis]

                if 'weights' not in detector.extra:
                    detector.extra['weights'] = []
                detector.extra['weights'].append(w)

                if 'sturf' not in detector.extra:
                    detector.extra['sturf'] = []

                detector.extra['sturf'].append(dict())
                        
                kernels.append(obj)
                bkgs.append(bkg)
                orig_sizes.append(size)
                new_support.append(alphas[m])

        detector.settings['per_mixcomp_bkg'] = True
    else:
        # Get a single background model for this one
        bkg = _get_background_model(settings, neg_files)

        argses = [(m, settings, list(np.where(comps == m)[0]), files) for m in range(detector.num_mixtures)]        
        for pos_feats in gv.parallel.starmap(_get_positives, argses):
            obj = pos_feats.mean(axis=0)
            all_pos_feats.append(pos_feats)

            kernels.append(obj)
            bkgs.append(bkg)
            size = gv.bb.size(bbs[m])

            orig_sizes.append(size)
            support = np.ones(settings['detector']['image_size'])
            new_support.append(support)

        detector.settings['per_mixcomp_bkg'] = True # False 


    # Get weights

    for m in xrange(detector.num_mixtures):
        #kern = detector.kernel_templates[m]
        #bkg = detector.fixed_spread_bkg[m]
        obj = all_pos_feats[m].mean(axis=0)
        bkg = all_neg_feats[m].mean(axis=0)

        if detector.eps is None:
            detector.prepare_eps(bkg)

        weights = detector.build_clipped_weights(obj, bkg, detector.eps)

        detector.extra['weights'][m] = weights


    # Modify weights
    if not detector.settings.get('plain'):
        for m in xrange(detector.num_mixtures):
            weights = detector.extra['weights'][m] 

            F = detector.num_features
            indices = get_key_points(weights, suppress_radius=detector.settings.get('indices_suppress_radius', 4), even=True)

            L0 = indices.shape[0] // F 
            
            kp_weights = np.zeros((L0, F))

            M = np.zeros(weights.shape, dtype=np.uint8)
            counts = np.zeros(F)
            for index in indices:
                f = index[2]
                M[tuple(index)] = 1
                kp_weights[counts[f],f] = weights[tuple(index)]
                counts[f] += 1

            #theta = np.load('theta3.npy')[1:-1,1:-1]
            #th = theta
            #eth = np.load('empty_theta.npy')

            #support = 1-th[:,:,np.arange(1,F+1),np.arange(F)].mean(-1)
            #offset = gv.sub.subsample_offset_shape(alphas[m].shape, psize)
            offset = tuple([(alphas[m].shape[i] - weights.shape[i] * psize[i])//2 for i in xrange(2)])

            support = gv.img.resize(alphas[m][offset[0]:offset[0]+psize[0]*weights.shape[0], \
                                              offset[1]:offset[1]+psize[1]*weights.shape[1]], weights.shape[:2]) 

            #    def subsample_offset_shape(shape, size):


            pos, neg = all_pos_feats[m].astype(bool), all_neg_feats[m].astype(bool)
            #avg = np.apply_over_axes(

            diff = pos ^ neg
            appeared = pos & ~neg
            disappeared = ~pos & neg

            #bs = (support > 0.5)[np.newaxis,...,np.newaxis]
             

            A = appeared.mean(0) / (0.00001+((1-neg).mean(0)))
            D = disappeared.mean(0) / (0.00001+neg.mean(0))
            #ss = D.mean(-1)[...,np.newaxis]
            ss = support[...,np.newaxis]

            B = (np.apply_over_axes(np.mean, A*ss, [0, 1])).squeeze() / ss.mean()

            def clogit(x):
                return gv.logit(gv.bclip(x, 0.025))

            def find_zero(fun, l, u, depth=30):
                m = np.mean([l, u])
                if depth == 0:
                    return m
                v = fun(m)
                if v > 0:
                    return find_zero(fun, l, m, depth-1)
                else:
                    return find_zero(fun, m, u, depth-1)

            # Find zero-crossing
            #for f in xrange(F):
                

            # Now construct weights from these deltas
            #weights = ((clogit(ss * deltas + A) - clogit(B)))
            #weights = (ss * (clogit(deltas + pos.mean(0)) - clogit(neg.mean(0))))

            
            avg = np.apply_over_axes(np.mean, pos * M * ss, [1, 2]) / (ss * M).mean()

            if 0:
                for l0, l1, f in gv.multirange(*weights.shape):

                    def fun(w):
                        return -(np.clip(pos[:,l0,l1,f].mean(), 0.005, 0.995) - np.mean(expit(w + logit(avg[...,f]))))

                    weights[l0,l1,f] = find_zero(fun, -10, 10)



            if 1:
                # Print these to file
                from matplotlib.pylab import cm
                grid = gv.plot.ImageGrid(detector.num_features, 1, weights.shape[:2], border_color=(0.5, 0.5, 0.5))
                mm = np.fabs(weights).max()
                for f in xrange(detector.num_features):
                    grid.set_image(weights[...,f], f, 0, vmin=-mm, vmax=mm, cmap=cm.RdBu_r)
                fn = os.path.join(os.path.expandvars('$HOME'), 'html', 'plots', 'plot2.png')
                grid.save(fn, scale=10)
                os.chmod(fn, 0644)
                



            #A = appeared.mean(0) / (0.00001+((1-neg).mean(0)))
            #mm = (A * ss).mean() / ss.mean()


            #xx = (bs & pos) | (~bs & appeared)

            #avg = xx.mean(0)
            weights1 = ss*(weights - np.apply_over_axes(np.mean, weights * ss, [0, 1])/ss.mean())
            detector.extra['sturf'][m]['weights1'] = weights1

            detector.extra['sturf'][m]['support'] = support

            eps = 0.025

            avg_pos = (np.apply_over_axes(np.mean, pos * ss, [0, 1, 2]) / ss.mean()).squeeze().clip(eps, 1-eps)
            avg_neg = (np.apply_over_axes(np.mean, neg * ss, [0, 1, 2]) / ss.mean()).squeeze().clip(eps, 1-eps)

            #w_avg = np.apply_over_axes(np.sum, weights * support[...,np.newaxis], [0, 1]) / support.sum()
            #
            #w_avg = (logit(np.apply_over_axes(np.mean, pos, [0, 1, 2])) - \
             #        logit(np.apply_over_axes(np.mean, neg, [0, 1, 2]))).squeeze()
            w_avg = logit(avg_pos) - logit(avg_neg)
            detector.extra['sturf'][m]['wavg'] = w_avg
            detector.extra['sturf'][m]['reweighted'] = (w_avg * support[...,np.newaxis]).squeeze()

            #weights -= w_avg * support[...,np.newaxis]
            #weights *= support[...,np.newaxis] * M
            if 0:
                weights *= support[...,np.newaxis]

                avg_weights = np.apply_over_axes(np.mean, weights, [0, 1]) / M.mean(0).mean(0)

                avg_w = kp_weights.mean(0)

                weights -= avg_w - (-kp_weights.var(0) / 2)

                weights *= support[...,np.newaxis]

                print((weights * M).mean(0))


            #weights = (weights - w_avg) * support[...,np.newaxis]
            #weights -= (w_avg + 0.0) * support[...,np.newaxis]

            weights -= w_avg * support[...,np.newaxis]

            F = detector.num_features

            if 0:
                for f in xrange(F):
                    #zz = np.random.normal(-1.5, size=(1, 1, 50))
                    zz = np.random.normal(-1.5, size=(1, 1, 50)).ravel()

                    betas = np.zeros(len(zz))
                    for i, z in enumerate(zz):
                        def fun(beta):
                            w = weights[...,f] - beta * support 
                            return np.log(1 - expit(w[...,np.newaxis] + z)).mean() - np.log(1 - expit(z))

                        betas[i] = find_zero(fun, -10, 10)

                    
                    if f == 0:
                        np.save('betas.npy', betas)
                    beta0 = betas.mean()
                    print(f, beta0, betas.std())
                    weights[...,f] -= beta0 * support 


            if 1:
                # Print these to file
                from matplotlib.pylab import cm
                grid = gv.plot.ImageGrid(detector.num_features, 2, weights.shape[:2], border_color=(0.5, 0.5, 0.5))
                mm = np.fabs(weights).max()
                for f in xrange(detector.num_features):
                    grid.set_image(weights[...,f], f, 0, vmin=-mm, vmax=mm, cmap=cm.RdBu_r)
                    grid.set_image(M[...,f], f, 1, vmin=0, vmax=1, cmap=cm.RdBu_r)
                fn = os.path.join(os.path.expandvars('$HOME'), 'html', 'plots', 'plot.png')
                grid.save(fn, scale=10)
                os.chmod(fn, 0644)

            print('sum', np.fabs(np.apply_over_axes(np.sum, weights, [0, 1])).sum())

            # Instead, train model rigorously!!
            detector.extra['sturf'][m]['pos'] = all_pos_feats[m]
            detector.extra['sturf'][m]['neg'] = all_neg_feats[m]


            # Averags of all positives
            ff = all_pos_feats[m]
            posavg = np.apply_over_axes(np.sum, all_pos_feats[m] * support[...,np.newaxis], [1, 2]).squeeze() / support.sum() 
            negavg = np.apply_over_axes(np.sum, all_neg_feats[m] * support[...,np.newaxis], [1, 2]).squeeze() / support.sum() 

            S = np.cov(posavg.T)
            Sneg = np.cov(negavg.T)

            detector.extra['sturf'][m]['pavg'] = avg_pos
            detector.extra['sturf'][m]['pos-samples'] = posavg 
            detector.extra['sturf'][m]['S'] = S
            detector.extra['sturf'][m]['Sneg'] = Sneg
            detector.extra['sturf'][m]['navg'] = avg_neg

            Spos = S
            rs = np.random.RandomState(0)
            detector.extra['sturf'][m]['Zs'] = rs.multivariate_normal(avg_neg, Sneg, size=1000).clip(min=0.005, max=0.995)
            detector.extra['sturf'][m]['Zs_pos'] = rs.multivariate_normal(avg_pos, Spos, size=1000).clip(min=0.005, max=0.995)
            detector.extra['sturf'][m]['Zs_pos2'] = rs.multivariate_normal(avg_pos, Spos * 2, size=1000).clip(min=0.005, max=0.995)
            detector.extra['sturf'][m]['Zs_pos10'] = rs.multivariate_normal(avg_pos, Spos * 10, size=1000).clip(min=0.005, max=0.995)
            detector.extra['sturf'][m]['Zs_pos50'] = rs.multivariate_normal(avg_pos, Spos * 50, size=1000).clip(min=0.005, max=0.995)

    #{{{
    if 0:
        argses = [(m, settings, bbs[m], np.where(comps == m)[0], files, neg_files) for m in xrange(detector.num_mixtures)]
        for kern, bkg, orig_size, sup in gv.parallel.starmap(_create_kernel_for_mixcomp, argses):
            kernels.append(kern) 
            bkgs.append(bkg)
            orig_sizes.append(orig_size)
            new_support.append(sup)
                    
            print("Checkpoint 10")

            detector.settings['per_mixcomp_bkg'] = True
    #}}}

    detector.kernel_templates = kernels
    detector.kernel_sizes = orig_sizes
    detector.settings['kernel_ready'] = True
    detector.use_alpha = False
    detector.support = new_support

    # Determine the background
    ag.info("Determining background")

    detector.fixed_bkg = None
    detector.fixed_spread_bkg = bkgs

    detector.settings['bkg_type'] = 'from-file'

    detector._preprocess()
    detector.prepare_eps(detector.fixed_spread_bkg[0])

    # Determine the standardization values
    ag.info("Determining standardization values")

    #fixed_train_mean = np.zeros(detector.num_mixtures)
    #detector.fixed_train_mean = []
    #fixed_train_std = np.ones(detector.num_mixtures)

    # Determine indices for coarse detection sweep
    if INDICES:
        detector.indices = []

        for m in xrange(detector.num_mixtures):
            these_indices = []
            weights = detector.extra['weights'][m]

            print('Indices:', np.prod(weights.shape))

            # If not plain, we need even keypoints
            even = not detector.settings.get('plain')
            indices = get_key_points(weights, suppress_radius=detector.settings.get('indices_suppress_radius', 4), even=even)

            if not detector.settings.get('plain'):
                detector.extra['weights'][m] = weights

            assert len(indices) > 0, "No indices were extracted when keypointing"

            detector.indices.append(indices)
    else:
        detector.indices = None

    if testing_type in ('fixed', 'non-parametric'):
        detector.standardization_info = []
        if testing_type == 'fixed':
            argses = [(m, settings, detector.eps, bbs[m], kernels[m], bkgs[m], None, None, None, detector.indices[m] if INDICES else None, 3) for m in xrange(detector.num_mixtures)]

            detector.standardization_info = list(gv.parallel.starmap(_calc_standardization_for_mixcomp, argses))
        else:
            raise Exception("Unknown testing type")


    detector.settings['testing_type'] = testing_type 
    #detector.settings['testing_type'] = 'NEW'

    #detector.

    #
    # Data mine stronger negatives 
    #
    # TODO: Object class must be input
    if 1:
        contest = 'voc'
        obj_class = 'car'
        gen = gv.voc.gen_negative_files(obj_class, 'train')
    else:
        contest = 'custom-tmp-frontbacks'
        obj_class = 'bicycle'
        gen, tot = gv.datasets.load_files(contest, obj_class)

    import heapq
    top_bbs = [[] for k in xrange(detector.num_mixtures)]
    TOP_N = 10000


    if detector.settings.get('cascade'): # New SVM attempt 
        detector.extra['cascade_threshold'] = detector.settings.get('cascade_threshold', 8) 
        COUNT = detector.settings.get('cascade_farming_count', 500)

        args = itr.izip( \
            itr.repeat(detector), 
            xrange(COUNT), 
            itr.islice(gen, COUNT)
        )

        for res in gv.parallel.starmap_unordered(get_strong_fps, args):
            for m in xrange(detector.num_mixtures):
                top_bbs[m].extend(res[m])

        print('- TOPS ------')
        print(map(np.shape, top_bbs) )
        detector.extra['top_bbs_shape'] = map(np.shape, top_bbs) 

        # Save the strong negatives
        detector.extra['negs'] = top_bbs
        
        def phi(X, mixcomp):
            if SVM_INDICES and 0:
                indices = detector.indices2[mixcomp][0]
                return X.ravel()[np.ravel_multi_index(indices.T, X.shape)]
            else:
                #return gv.sub.subsample(X, (2, 2)).ravel()
                return X.ravel()

        all_neg_X0 = []
        for k in xrange(detector.num_mixtures):
            all_neg_X0.append(np.asarray(map(lambda bbobj: phi(bbobj.X, k), top_bbs[k])))

        del top_bbs

        # OLD PLACE FOR FETCHING POSITIVES
        if 0:
            # Retrieve positives
            ag.info('Fetching positives again...')
            argses = [(m, settings, bbs[m], list(np.where(comps == m)[0]), files, neg_files, settings['detector'].get('stand_multiples', 1)) for m in range(detector.num_mixtures)]        
            all_pos_feats = list(gv.parallel.starmap(_get_positives, argses))
            all_pos_X0 = []
            for mixcomp, pos_feats in enumerate(all_pos_feats):
                all_pos_X0.append(np.asarray(map(lambda X: phi(X, mixcomp), pos_feats))) 
            ag.info('Done.')

        all_pos_X0 = []
        for mixcomp, pos_feats in enumerate(all_pos_feats):
            all_pos_X0.append(np.asarray(map(lambda X: phi(X, mixcomp), pos_feats))) 
        ag.info('Done.')

        detector.extra['poss'] = all_pos_feats

        ag.info('Training SVMs...')
        # Train SVMs
        #from sklearn.svm import LinearSVC
        from sklearn.svm import LinearSVC, SVC
        clfs = []
        detector.indices2 = None # not [] for now 

        #all_neg_X0 = [[bbobj.X for bbobj in top_bbs[m]] for m in xrange(detector.num_mixtures)]

        detector.extra['svms'] = []
        for m in xrange(detector.num_mixtures):
            try:
                X = np.concatenate([all_pos_X0[m], all_neg_X0[m]])  
            except:
                import pdb; pdb.set_trace()
    
            # Flatten
            print(m, ':', X.shape)
            #X = phi(X, k)
            print(m, '>', X.shape)
            y = np.concatenate([np.ones(len(all_pos_feats[m])), np.zeros(len(all_neg_X0[m]))])

            #detector.extra['data_x'].append(X)
            #detector.extra['data_y'].append(y)


            from sklearn import cross_validation as cv

            #C = 5e-8
            C = 1.0

            #clf = LinearSVC(C=C)
            #clf = LinearSVC(C=C)
            clf = SVC(C=C, kernel='linear')
            clf.fit(X, y)

            svm_info = dict(intercept=float(clf.intercept_), weights=clf.coef_)
            detector.extra['svms'].append(svm_info)

            #sh = all_pos_feats[m][0].shape

            # Get most significant coefficients

            #th = smallest_th[k] 
            #th = 0
            #detector.extra['svms'].append(dict(svm=clf, th=th, uses_indices=SVM_INDICES))
        ag.info('Done.')

        # Remove negatives and positives from extra, since it takes space
        if 1:
            del detector.extra['poss']
            del detector.extra['negs']

    print('extra')
    print(detector.extra.keys())
    print('eps', detector.eps)

    #print("THIS IS SO TEMPORARY!!!!!")
    if 'weights' in detector.extra:
        #detector.indices = None

        print(detector.standardization_info)
        #try:
        #    detector.standardization_info[0]['std'] = 1.0
        #except TypeError:
        #    detector.standardization_info = [dict(std=1.0, mean=0.0)]
        print('corner2', detector.extra['weights'][0][0,0,:5])

    return detector 


ag.set_verbose(True)
if gv.parallel.main(__name__): 
    import argparse
    from settings import load_settings
        
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
