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
from gv.keypoints import get_key_points_even

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


def _create_kernel_for_mixcomp_star(args):
    return _create_kernel_for_mixcomp(*args)


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


def _calc_standardization_for_mixcomp_star(args):
    return _calc_standardization_for_mixcomp(*args)


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

def _get_positives_star(args):
    return _get_positives(*args)

def __process_bkg(fn, descriptor, sett, factor):
    im = gv.img.asgray(gv.img.load_image(fn))
    im = gv.img.resize_with_factor_new(im, factor)


    ag.info("Processing image for background model:", fn)
    feats = descriptor.extract_features(im, settings=sett)

    #count += np.prod(feats.shape[:2])
    #bkg_counts += np.apply_over_axes(np.sum, feats, [0, 1]).ravel()
    return np.apply_over_axes(np.sum, feats, [0, 1]).ravel(), np.prod(feats.shape[:2])

def __process_bkg_star(args):
    return __process_bkg(*args)

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

    for feats, c in gv.parallel.imap_unordered(__process_bkg_star, argses):
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

def _process_file_kernel_basis_star(args):
    return _process_file_kernel_basis(*args)


def __process_one(args):
    index, mixcomp, files, im_size, bb, duplicates, neg_files, descriptor, sett = args
    size = gv.bb.size(bb)
    psize = sett['subsample_size']

    ADAPTIVE = False 
    if ADAPTIVE:
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

def _get_pos_and_neg_star(args):
    return _get_pos_and_neg(*args)



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

def get_strong_fps_star(args):
    return get_strong_fps(*args)


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


    # Get basis
    if 0:
        data = np.load('bkg-stack-np.npz')
        bkg_stack = data['bkg_stack']
        bkg_stack_num = data['bkg_stack_num']
        for m in xrange(detector.num_mixtures):
            files_m = [files[ii] for ii in np.where(comps == m)[0]]

            all_counts = None
            all_empty_counts = None
            all_totals = None

            argses = [(seed, m, settings, bbs[m], fn, bkg_stack, bkg_stack_num) for seed, fn in enumerate(files_m)]
            for counts, empty_counts, totals in gv.parallel.imap_unordered(_process_file_kernel_basis_star, argses):
                if all_counts is None:
                    all_counts = counts
                    all_empty_counts = empty_counts
                    all_totals = totals
                else:
                    all_counts += counts
                    all_empty_counts += empty_counts
                    all_totals += totals

            ag.info('Done.')

            np.savez('atheta.npz', theta=all_counts.astype(np.float64) / all_totals, 
                                   empty_theta=all_empty_counts.astype(np.float64) / all_totals)
            #np.save('theta2.npy', all_counts.astype(np.float64) / all_totals)
            #np.save('empty_theta.npy', all_empty_counts.astype(np.float64) / all_totals)
            #np.savez('theta2.npy', counts=all_counts.astype(np.float64) / all_totals, empty_counts=all_empty_counts.astype(np.float64) / all_totals)
            import sys; sys.exit(1)

        #argses = [(m, settings, bbs[m], list(np.where(comps == m)[0]), files, neg_files, settings['detector'].get('stand_multiples', 1), bkg_stack) for m in range(detector.num_mixtures)]        
        #for counts, totals in gv.parallel.imap(_get_kernel_basis_star, argses):
            #alpha = alpha_maps.mean(0)
            #all_alphas.append(alpha_maps)
            #all_binarized_alphas.append(alpha_maps > 0.05)
#
            #alphas.append(alpha)
            #all_neg_feats.append(neg_feats)
            #all_pos_feats.append(pos_feats)
#
        ag.info('Done.')

        np.save('theta.npy', theta)
        import sys; sys.exit(0)


    if settings['detector'].get('superimpose'):
        argses = [(m, settings, bbs[m], list(np.where(comps == m)[0]), files, neg_files, settings['detector'].get('stand_multiples', 1)) for m in range(detector.num_mixtures)]        
        #for neg_feats, pos_feats, alpha_maps in gv.parallel.imap(_get_pos_and_neg_star, argses):
        for neg_feats, pos_feats, alpha_maps in itr.imap(_get_pos_and_neg_star, argses):
            alpha = alpha_maps.mean(0)
            all_alphas.append(alpha_maps)
            all_binarized_alphas.append(alpha_maps > 0.05)

            alphas.append(alpha)
            all_neg_feats.append(neg_feats)
            all_pos_feats.append(pos_feats)

        ag.info('Done.')

        for m in xrange(detector.num_mixtures):

            #feats = np.concatenate([all_pos_feats[m], all_neg_feats[m]], axis=0)
            #labels = np.zeros(len(all_pos_feats[m]) + len(all_neg_feats[m]))
            #labels[:len(all_pos_feats)] = 1
            #np.savez('feats.npz', feats=feats, labels=labels)

            # Find key points


            if 1:
                obj = all_pos_feats[m].mean(axis=0)
                bkg = all_neg_feats[m].mean(axis=0)
                size = gv.bb.size(bbs[m])

                avg_frames = np.apply_over_axes(np.mean, all_pos_feats[m], [1, 2]).squeeze()
                means = avg_frames.mean(0)
                stds = avg_frames.std(0)

                #small_alpha_maps = (gv.sub.subsample(all_alphas[m][:,2:-2,3:-3], (4, 4), skip_first_axis=True) > 0.05)[...,np.newaxis].astype(np.uint8)

                #for i in xrange(small_alpha_maps.shape[0]):
                    #small_alpha_maps[i] = ag.util.blur_image(small_alpha_maps[i], 2.0)
                    #small_alpha_maps[i,...,0] = ag.features.bspread(small_alpha

                # SET WEIGHTS
                #avg = np.apply_over_axes(np.mean, all_pos_feats[m], [0, 1, 2])[0]
                #bkg = avg
                avg = bkg
                #avg = np.apply_over_axes(np.mean, all_pos_feats[m][:,2:-2,2:-2], [0, 1, 2])[0]
                #avg = np.apply_over_axes(np.sum, small_alpha_maps * all_pos_feats[m], [0, 1, 2])[0] / np.sum(small_alpha_maps)
                #import pdb; pdb.set_trace()

                #small_alpha_maps[...,0] = ag.features.bspread(small_alpha_maps[...,0], spread='box', radius=1)

                #obj = small_alpha_maps * obj + ~small_alpha_maps * avg
                #obj = (all_pos_feats[m] * small_alpha_maps).mean(0) + (avg * (1 - small_alpha_maps)).mean(0)

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

                detector.extra['sturf'].append(dict(means=means, stds=stds, lmb=obj / avg))


                    
            #{{{
            else:
                # This is experimental code:

                # Train SVM and get negative support vectors
                # Figure out covariance matrix here, since we have neg and pos
                if 0:
                    try:
                        L = np.prod(all_pos_feats[m].shape[1:])
                        Npos = all_pos_feats[m].shape[0]
                        from sklearn.svm import SVC
                        svm = SVC(kernel='linear', C=1)
                        X = np.concatenate([all_pos_feats[m].reshape((-1, L)), 
                                            all_neg_feats[m].reshape((-1, L))])
                        y = np.zeros(X.shape[0])
                        y[:Npos] = 1
                        print("Training SVM")
                        svm.fit(X, y) 
                        print("Done")
                
                        II = svm.support_[svm.support_ >= Npos] - Npos
                        all_neg_feats[m] = all_neg_feats[m][II]
                    except:
                        import pdb; pdb.set_trace()

                obj = all_pos_feats[m].mean(axis=0)
                bkg = all_neg_feats[m].mean(axis=0)
                size = gv.bb.size(bbs[m])

                #import pdb; pdb.set_trace()

                sh = obj.shape
                L = np.prod(sh)

                avg = np.zeros(obj.shape)
                kern = 0.5 * np.ones(obj.shape) 
                # Calculate kernel using LDA
                p1 = all_pos_feats[m].reshape((-1, L))
                n1 = all_neg_feats[m].reshape((-1, L))
                p1mean = p1.mean(0)
                n1mean = n1.mean(0)

                Spos = reduce(np.add, (np.outer(*[xi - p1mean]*2) for xi in p1)) / p1.shape[0]
                Sneg = reduce(np.add, (np.outer(*[xi - n1mean]*2) for xi in n1)) / n1.shape[0]

                #S = Sneg + Spos
                S = Sneg
                #S[:] += np.random.normal(loc=0, scale=0.001, size=S.shape)
                # Regularization (important!)
                np.save('S.npy', S)

                S[:] += np.eye(S.shape[0]) * 0.01
                #Sinv = np.linalg.inv(S)
                #Sinv = np.eye(S.shape[0])
                #kern[i,j] = np.dot(Sinv, (p1mean - n1mean))
                #kern[:] = np.dot(Sinv, (p1mean - n1mean)).reshape(sh)

                #kern[:] = np.linalg.solve(S, p1mean - n1mean).reshape(sh)
                np.save('p1mean.npy', p1mean)
                np.save('n1mean.npy', n1mean)

                #detector.build_clipped_weights(obj, bkg, )

                kern[:] = np.linalg.solve(S, np.ones(p1mean.shape)).reshape(sh)

                w = kern[:]
                II = np.argsort(np.fabs(w.ravel()))
                M = np.zeros(w.shape)
                for rank, ii in enumerate(II):
                    M[tuple(np.unravel_index(ii, w.shape))] = rank
                np.save('M.npy', M)

                #import IPython
                #IPython.embed()

                if 0:
                    for i, j in itr.product(xrange(obj.shape[0]), xrange(obj.shape[1])):
                        p1 = all_pos_feats[m][:,i,j]
                        n1 = all_neg_feats[m][:,i,j]
                        p1mean = p1.mean(0)
                        n1mean = n1.mean(0)

                        Spos = np.mean([np.outer(*[xi - p1mean]*2) for xi in p1], axis=0) 
                        Sneg = np.mean([np.outer(*[xi - n1mean]*2) for xi in n1], axis=0) 

                        S = Sneg + Spos
                        try:
                            Sinv = np.linalg.pinv(S)
                            #Sinv = np.eye(S.shape[0])
                            kern[i,j] = np.dot(Sinv, (p1mean - n1mean))
                        except:
                            print('Skipping', i, j)
                #bkg[:] = 0.5

                detector.extra['weights'] = kern
                    #avg[:,i,j] = (p1mean + n1mean) / 2

                #self.extra['avg'] = avg
            #}}}

            kernels.append(obj)
            bkgs.append(bkg)
            orig_sizes.append(size)
            new_support.append(alphas[m])

        detector.settings['per_mixcomp_bkg'] = True
    else:
        # Get a single background model for this one
        bkg = _get_background_model(settings, neg_files)

        argses = [(m, settings, list(np.where(comps == m)[0]), files) for m in range(detector.num_mixtures)]        
        for pos_feats in gv.parallel.imap(_get_positives_star, argses):
            obj = pos_feats.mean(axis=0)
            all_pos_feats.append(pos_feats)

            kernels.append(obj)
            bkgs.append(bkg)
            size = gv.bb.size(bbs[m])

            orig_sizes.append(size)
            support = np.ones(settings['detector']['image_size'])
            new_support.append(support)

        detector.settings['per_mixcomp_bkg'] = True # False 

    #{{{
    if 0:
        argses = [(m, settings, bbs[m], np.where(comps == m)[0], files, neg_files) for m in xrange(detector.num_mixtures)]
        for kern, bkg, orig_size, sup in gv.parallel.imap(_create_kernel_for_mixcomp_star, argses):
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

            kern = detector.kernel_templates[m]
            bkg = detector.fixed_spread_bkg[m]
            if detector.eps is None:
                detector.prepare_eps(bkg)

            #kern = np.clip(kern, detector.eps, 1 - eps)
            #bkg = np.clip(bkg, eps, 1 - eps)
            #weights = np.log(kern / (1 - kern) * ((1 - bkg) / bkg))
            weights = detector.build_clipped_weights(kern, bkg, detector.eps)

            if 1:
                F = detector.num_features

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


                if 0:
                    deltas = np.zeros(F)
                    for f in xrange(F):
                        def fun(x):
                            #return ((clogit(ss[...,0] * x + A[...,f]) - clogit(B[...,f]))).mean(0).mean(0)
                            return (ss[...,0] * (clogit(x + pos[...,f].mean(0)) - clogit(neg[...,f].mean(0)))).mean(0).mean(0)
                        deltas[f] = find_zero(fun, -5, 5)

                # Now construct weights from these deltas
                #weights = ((clogit(ss * deltas + A) - clogit(B)))
                #weights = (ss * (clogit(deltas + pos.mean(0)) - clogit(neg.mean(0))))


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

                avg_pos = (np.apply_over_axes(np.mean, pos * ss, [0, 1, 2]) / ss.mean()).squeeze()
                avg_neg = (np.apply_over_axes(np.mean, neg * ss, [0, 1, 2]) / ss.mean()).squeeze()

                from scipy.special import logit, expit as sigmoid
                #w_avg = np.apply_over_axes(np.sum, weights * support[...,np.newaxis], [0, 1]) / support.sum()
                #
                #w_avg = (logit(np.apply_over_axes(np.mean, pos, [0, 1, 2])) - \
                 #        logit(np.apply_over_axes(np.mean, neg, [0, 1, 2]))).squeeze()
                w_avg = logit(avg_pos) - logit(avg_neg)
                detector.extra['sturf'][m]['wavg'] = w_avg
                detector.extra['sturf'][m]['reweighted'] = (w_avg * support[...,np.newaxis]).squeeze()

                #import pdb; pdb.set_trace()

                #weights -= w_avg * support[...,np.newaxis]
                #weights *= support[...,np.newaxis]
                #weights = (weights - w_avg) * support[...,np.newaxis]
                weights -= w_avg * support[...,np.newaxis]

                if 1:
                    # Print these to file
                    from matplotlib.pylab import cm
                    grid = gv.plot.ImageGrid(detector.num_features, 1, weights.shape[:2], border_color=(0.5, 0.5, 0.5))
                    mm = np.fabs(weights).max()
                    for f in xrange(detector.num_features):
                        grid.set_image(weights[...,f], f, 0, vmin=-mm, vmax=mm, cmap=cm.RdBu_r)
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
                #import pdb; pdb.set_trace()

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

            if 0:
                #{{{c
                    # Set weights! 
                    theta = np.load('theta3.npy')[1:-1,1:-1]
                    th = theta
                    eth = np.load('empty_theta.npy')

                    #rest = 0.5

                    F = detector.num_features
                    bkg = 0.005 * np.ones(F)
                    #rest = 1 - bkg.sum()
                    if bkg.sum() <= 1:
                        rest = 1 - bkg.sum() 
                    else:
                        rest = 0
                    plus_bkg = np.concatenate([[rest], bkg])
                    plus_bkg /= plus_bkg.sum()

                    obj = np.zeros(weights.shape)

                    import scipy.optimize as opt

                    #{{{ 
                    if 0: 
                        global it
                        it = 0
                        def tick(plus_bkg):
                            global it
                            #plus_bkg = np.clip(plus_bkg, 0, np.inf)
                            #plus_bkg /= plus_bkg.sum()

                            for i, j in itr.product(xrange(weights.shape[0]), xrange(weights.shape[1])):
                                obj[i,j] = np.dot(theta[i,j].T, plus_bkg)
                            new_bkg = np.apply_over_axes(np.mean, obj, [0, 1]).squeeze()
                            corner_bkg = np.dot(eth.T, plus_bkg)#[0,0]
                            #corner_bkg = theta[i,j,0]
                            
                            it += 1
                            diff = np.sum(np.fabs(corner_bkg - new_bkg))
                            print('iteration', it, 'diff', diff, 'x sum', plus_bkg.sum(), 'x min', plus_bkg.min())

                        def const_f(plus_bkg):
                            return plus_bkg.sum() - 1.0

                        #def const_g(plus_bkg):
                            #return plus_bkg.min()

                        def fun(plus_bkg):
                            plus_bkg /= plus_bkg.sum()
                            for i, j in itr.product(xrange(weights.shape[0]), xrange(weights.shape[1])):
                                obj[i,j] = np.dot(theta[i,j].T, plus_bkg)
                            new_bkg = np.apply_over_axes(np.mean, obj, [0, 1]).squeeze()
                            #corner_bkg = obj[0,0]
                            #corner_bkg = obj[0,0]
                            corner_bkg = np.dot(eth.T, plus_bkg)#[0,0]
                            #corner_bkg = theta[i,j,0]

                            eps = 0.01
                            
                            objc = np.clip(obj, eps, 1 - eps)
                            avgc = np.clip(corner_bkg, eps, 1 - eps)
                            w = np.log(objc / (1 - objc) * (1 - avgc) / avgc)
                            #return np.sum(np.fabs(corner_bkg - new_bkg)**2)
                            return (w.mean(0).mean(0)**2).sum()

                        const = [dict(type='eq', fun=const_f)#, dict(type='ineq', fun=const_g)]
                                ]

                        def iv(n, i):
                            x = np.zeros(n)
                            x[i] = 1
                            return x

                    if 0:

                        res = opt.minimize(fun, plus_bkg, method='SLSQP', options=dict(maxiter=250), constraints=const, bounds=[(0, 1)] * plus_bkg.size)
                        plus_bkg = res['x']
                        #plus_bkg /= plus_bkg.sum()

                    elif 0:
                        N = 10000
                        #res = opt.minimize(f, plus_bkg, method='SLSQP', options=dict(maxiter=250), constraints=const, bounds=[(0, 1)] * plus_bkg.size)
                        min_score = np.inf
                        for i in xrange(N):
                            #f = i % (plus_bkg.size)

                            #delta = iv(plus_bkg.size, rs.randint(plus_bkg.size)) * rs.normal(scale=scale)
                            #x = plus_bkg + rs.normal(loc=0, scale=0.0001, size=plus_bkg.size)
                            #x /= x.sum()

                            rs = np.random.RandomState(i)
                            if 0:
                                #x = rs.uniform(0, 1, size=plus_bkg.size)
                                if 0:
                                    scale = 0.01
                                    if i > 10000:
                                        scale = 0.001
                                    elif i > 30000:
                                        scale = 0.0001

                                d = iv(plus_bkg.size, f)
                                @np.vectorize
                                def evaluate(c):
                                    xx = plus_bkg + d * c
                                    xx /= xx.sum()
                                    return fun(xx)

                                # Do a bisection
                                if 0:
                                    c = np.linspace(-0.1, 0.1, 11)
                                    scores = evaluate(c)
                                    mi = np.argmin(scores)

                                plus_bkg[:] = plus_bkg + d * c[mi]
                                plus_bkg /= plus_bkg.sum()
                                min_score = scores[mi]
                                print(i, min_score)
                            elif 1: 
                                #delta = iv(plus_bkg.size, rs.randint(plus_bkg.size)) * rs.normal(scale=scale)
                                #x = plus_bkg + delta 
                                x = plus_bkg + rs.normal(loc=0, scale=0.001, size=plus_bkg.size)
                                x = np.clip(x, 0, 1)
                                x /= x.sum()
                                s0, s1 = fun(x), 0# 10000 * np.sum((x - 1/plus_bkg.size)**2)
                                s = s0 + s1
                                if i % 50 == 0:
                                    print(i, min_score, s, s0, s1, plus_bkg[[0,50,197,198]])
                                if s < min_score:
                                    plus_bkg[:] = x 
                                    min_score = s
                    #}}}
                    #plus_bkg = scores[np.argmin(scores)]

                    N = 2500
                    

                    if 0:
                        sb = np.zeros((N, F))
                        ss = np.zeros((N,) + weights.shape)

                        min_score = np.inf * np.ones(F)
                        best_x = np.zeros(F)
                          
                        for i in xrange(N):
                            rs = np.random.RandomState(i)
                            x = rs.uniform(0, 1, size=plus_bkg.size)
                            x[rs.randint(F)+1] += rs.randint(100)
                            x[0] += rs.randint(200)
                            x /= x.sum()
                            bkg = (eth * x[...,np.newaxis]).sum(0)

                            obj = (th * x[np.newaxis,np.newaxis,...,np.newaxis]).sum(2) 

                            #avg = np.apply_over_axes(np.mean, abj, [0, 1]).squeeze()

                            sb[i] = bkg 
                            ss[i] = obj 

                        avgs = np.apply_over_axes(np.mean, ss, [1, 2]).squeeze()
                        diff = np.fabs(avgs - sb)

                        II = diff.argmin(0)

                        corner_bkg = np.diag(sb[II])
                        obj = np.rollaxis(ss[II][np.arange(F),...,np.arange(F)], 0, 3)

                    x = np.ones(F + 1)
                    x /= x.sum()

                    corner_bkg = (eth * x[...,np.newaxis]).sum(0)
                    obj = (th * x[np.newaxis,np.newaxis,...,np.newaxis]).sum(2) 

                    #obj = np.rollaxis(ss[II][np.arange(F),...,np.arange(F)], 0, 3)

                    #import pdb; pdb.set_trace()

                        #eps = 0.001
                        #objc = np.clip(obj, eps, 1 - eps)
                        #bkgc = np.clip(bkg, eps, 1 - eps)

                        #avg = np.apply_over_axes(np.mean, abj, [0, 1]).squeeze()

                        #w = np.log(objc / bkgc), np.log((1 - objc) / (1 - bkgc))

                    #print(res)
                    #plus_bkg = res['x']
                    #plus_bkg /= plus_bkg.sum()
                    if 0:
                        for i, j in itr.product(xrange(weights.shape[0]), xrange(weights.shape[1])):
                            obj[i,j] = np.dot(theta[i,j].T, plus_bkg)

                        #corner_bkg = obj[0,0]
                        corner_bkg = np.dot(eth.T, plus_bkg)#[0,0]

                    if 0:
                        for i in xrange(10000):
                            for i, j in itr.product(xrange(weights.shape[0]), xrange(weights.shape[1])):
                                obj[i,j] = np.dot(theta[i,j].T, plus_bkg)

                            #avg = bkg
                            new_bkg = np.apply_over_axes(np.mean, obj, [0, 1]).squeeze()
                            corner_bkg = obj[0,0]

                            diff = (corner_bkg - new_bkg)

                            d = np.fabs(diff).sum()
                            print('total abs diff', d)

                            new_plus_bkg = np.concatenate([[0], new_bkg])

                            step = (diff > 0) * 2 - 1
                            #plus_bkg = 
                            plus_bkg[1:] -= 0.000001 * step
                            plus_bkg[0] += (0.000001 * step).sum()

                            plus_bkg /= plus_bkg.sum() 

                            plus_bkg = np.clip(plus_bkg, 0.0000001, 1.0)

                            #new_bkg /= new_bkg.sum()
                            #new_plus_bkg = np.concatenate([[0], new_bkg])
                            #print('diff', np.fabs(plus_bkg - new_plus_bkg).sum())
                            #plus_bkg = new_plus_bkg

                    eps = 0.025
                    
                    obj = np.clip(obj, eps, 1 - eps)
                    #avg = np.clip(avg, eps, 1 - eps)
                    avg = np.clip(corner_bkg, eps, 1 - eps)

                    #weights = np.log(obj / (1 - obj) * (1 - avg) / avg)

                    # TODO: Adjust
                    #for f in xrange(F):

                    support = 1-th[:,:,np.arange(1,F+1),np.arange(F)].mean(-1)
                    detector.extra['sturf'][m]['support'] = support

                    if 0:
                        w_avg = np.apply_over_axes(np.sum, weights * support[...,np.newaxis], [0, 1]) / support.sum()

                        weights -= w_avg * support[...,np.newaxis]


                    print('Indices:', np.prod(weights.shape))
                    indices = get_key_points_even(weights, suppress_radius=detector.settings.get('indices_suppress_radius', 4))
                    print('After local suppression:', indices.shape[0])
                        
                    #if 'weights' not in detector.extra:
                        #detector.extra['weights'] = []
                    detector.extra['weights'][m] = weights

                    detector.extra['sturf'][m]['means2'] = corner_bkg
                    detector.extra['sturf'][m]['bkg'] = np.load('uiuc-bkg.npy') # Obviously TODO

                    print('corner', detector.extra['weights'][0][0,0,:5])

                    if 0:
                        pos = all_pos_feats[m].reshape((all_pos_feats[m].shape[0], -1))
                        neg = all_neg_feats[m].reshape((all_neg_feats[m].shape[0], -1))

                        IG = np.zeros(len(indices))
                        for i, index in enumerate(indices):
                            px1 = (all_pos_feats[m][:,index[0], index[1], index[2]].mean() + all_neg_feats[m][:,index[0], index[1], index[2]].mean()) / 2

                            neg_f0 = (all_neg_feats[m][:,index[0], index[1], index[2]] == 0)
                            pos_f0 = (all_pos_feats[m][:,index[0], index[1], index[2]] == 0)

                            neg_f1 = (all_neg_feats[m][:,index[0], index[1], index[2]] == 1)
                            pos_f1 = (all_pos_feats[m][:,index[0], index[1], index[2]] == 1)

                            eps = 1e-5
                            neg_f0mean = np.clip(neg_f0.mean(), eps, 1 - eps)
                            neg_f1mean = np.clip(neg_f1.mean(), eps, 1 - eps)
                            pos_f0mean = np.clip(pos_f0.mean(), eps, 1 - eps)
                            pos_f1mean = np.clip(pos_f1.mean(), eps, 1 - eps)

                            h_xf0 = -(neg_f0mean * np.log2(neg_f0mean) + pos_f0mean * np.log2(pos_f0mean))
                            h_xf1 = -(neg_f1mean * np.log2(neg_f1mean) + pos_f1mean * np.log2(pos_f1mean))

                            IG[i] = h_xf0 * (1 - px1) + h_xf1 * px1 

                        from scipy.stats import scoreatpercentile
                        th = scoreatpercentile(IG, 50)

                        ok = (IG >= th)

                        indices = indices[ok]

                        print('After IG suppression:', indices.shape[0])
                    #}}}
            elif 1:
                
                print('Indices:', np.prod(weights.shape))

                indices = get_key_points_even(weights, suppress_radius=detector.settings.get('indices_suppress_radius', 4))

                detector.extra['weights'][m] = weights
            else:
                #{{{
                print('Indices:', np.prod(weights.shape))

                indices = get_key_points_even(weights, suppress_radius=detector.settings.get('indices_suppress_radius', 4))

                print('After local suppression:', indices.shape[0])

                raveled_indices = np.asarray([np.ravel_multi_index(index, weights.shape) for index in indices])


                
                #raveled_indices = np.arange(np.prod(weights.shape))

                pos = all_pos_feats[m].reshape((all_pos_feats[m].shape[0], -1))
                neg = all_neg_feats[m].reshape((all_neg_feats[m].shape[0], -1))

                feats = np.concatenate([pos[:,raveled_indices], neg[:,raveled_indices]])
                #feats = np.concatenate([all_pos_feats[m], all_neg_feats[m]])
                labels = np.zeros(feats.shape[0])
                labels[:len(all_pos_feats[m])] = 1

                # Train a sparse SVM
                from sklearn.svm import LinearSVC
                from sklearn.linear_model import LogisticRegression
                ag.info("Training L1 classifier for keypointing")
                #cl = LinearSVC(C=10000000.0, penalty='l1', dual=False)
                if 1:
                    cl = LogisticRegression(C=100000.0, penalty='l1', dual=False, tol=0.00001)
                else:
                    cl = LinearSVC(C=1.0)
                ag.info("Done")
                cl.fit(feats.reshape((feats.shape[0], -1)), labels)

                II = np.where(np.fabs(cl.coef_.ravel()) >= 0.001)[0]


                if 0:
                    w = cl.coef_.reshape(weights.shape)
                    if 'weights' not in detector.extra:
                        detector.extra['weights'] = []
                    detector.extra['weights'].append(w)

                indices = indices[II]

                print('Final indices:', indices.shape[0])
                #new_indices = []
                #for i, j, k in itr.product(xrange(feats.shape[1]), xrange(feats.shape[2]), xrange(feats.shape[3])):

                    #if np.fabs(coef[i,j,k]) > 0.001:
                        #new_indices.append((i,j,k))        

                indices = np.asarray(indices)


                #raw_input('Press...')
                #}}}

            assert len(indices) > 0, "No indices were extracted when keypointing"

            detector.indices.append(indices)
    else:
        detector.indices = None

    if testing_type in ('fixed', 'non-parametric'):
        detector.standardization_info = []
        if testing_type == 'fixed':
            argses = [(m, settings, detector.eps, bbs[m], kernels[m], bkgs[m], None, None, None, detector.indices[m] if INDICES else None, 3) for m in xrange(detector.num_mixtures)]

            detector.standardization_info = list(gv.parallel.imap(_calc_standardization_for_mixcomp_star, argses))
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

        for res in gv.parallel.imap_unordered(get_strong_fps_star, args):
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
            all_pos_feats = list(gv.parallel.imap(_get_positives_star, argses))
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
