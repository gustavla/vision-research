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
import itertools
from collections import namedtuple
from superimpose_experiment import generate_random_patches

K = 8 

#Patch = namedtuple('Patch', ['filename', 'selection'])

#def load_patch_image(patch):
#    img = gv.img.asgray(gv.img.load_image(patch.filename))
#    return img[patch.selection]

def generate_random_patches(filenames, size, seed=0, per_image=1):
    randgen = np.random.RandomState(seed)
    failures = 0
    for fn in itertools.cycle(filenames):
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

def fetch_bkg_model(settings, neg_files):
    randgen = np.random.RandomState(0)

    size = settings['detector']['image_size']
    descriptor = gv.load_descriptor(settings)

    radii = settings['detector']['spread_radii']
    psize = settings['detector']['subsample_size']
    cb = settings['detector'].get('crop_border')

    counts = np.zeros(descriptor.num_features)
    tot = 0

    for fn in neg_files[:200]:
        ag.info('Processing {0} for background model extraction'.format(fn))

        im = gv.img.resize_with_factor_new(gv.img.asgray(gv.img.load_image(fn)), randgen.uniform(0.5, 1.0))

        subfeats = descriptor.extract_features(im, settings=dict(spread_radii=radii, subsample_size=psize, crop_border=cb))
        #subfeats = gv.sub.subsample(feats, psize)
        x = np.rollaxis(subfeats, 2).reshape((descriptor.num_features, -1))
        tot += x.shape[1]
        counts += x.sum(axis=1)
        
    return counts / tot 
    

def _partition_bkg_files(seed, count, settings, size, neg_files, files, bb):
    im_size = settings['detector']['image_size'] # TEMP

    gen = generate_random_patches(neg_files, size, seed=0)

    descriptor = gv.load_descriptor(settings)

    radii = settings['detector']['spread_radii']
    psize = settings['detector']['subsample_size']
    cb = settings['detector'].get('crop_border')
    setts = dict(spread_radii=radii, subsample_size=psize, crop_border=cb)

    np.apply_over_axes(np.mean, descriptor.parts, [1, 2]).reshape((-1, 4))
    
    orrs = np.apply_over_axes(np.mean, descriptor.parts, [1, 2]).reshape((-1, 4))
    norrs = orrs / np.expand_dims(orrs.sum(axis=1), 1)
       
    neg_ims = []
    feats = []
    use = np.ones(count, dtype=bool)

    #prnd = np.random.RandomState(1)
    # TEMP
    #return [], np.clip(prnd.normal(loc=0.05, scale=0.05, size=(K, descriptor.num_features)), 0.0, 1.0)

    for i, neg_im in enumerate(gen):
        if i == count:
            break

        gray_im, alpha = _load_cad_image(files[i%len(files)], im_size, bb)
        superimposed_im = neg_im * (1 - alpha) + gray_im * alpha

        #im = superimposed_im
        im = neg_im

        if i % 100 == 0:
            ag.info("Loading bkg im {0}".format(i))
    
        all_feat = descriptor.extract_features(im, settings=setts)
        feat = np.apply_over_axes(np.mean, all_feat, [0, 1]).ravel() 

        if 0:
            featsum = feat.sum()
            if featsum == 0:
                #feat = np.ones_like(feat)
                use[i] = False 

            else: 
                #feat /= featsum
                #orientations = (np.expand_dims(feat, 1) * norrs).mean(axis=0)
                #feats.append(orientations)
                pass
        feats.append(feat)

        #neg_ims.append(neg_im)

    feats = np.asarray(feats)
    #neg_ims = np.asarray(neg_ims)

    print "Starting GMM"
    #from sklearn.mixture import GMM, KMeans
    from sklearn.cluster import KMeans
    if K > 1:
        #mixture = GMM(K)
        mixture = KMeans(K)
        print "Created GMM"
        mixture.fit(feats)
        print "Fit GMM"

        mixcomps = mixture.predict(feats)
    else:
        mixcomps = np.zeros(len(feats))

    print "Predicted"

    #return [neg_ims[mixcomps==i] for i in xrange(K)]
    if 0:
        full_mixcomps = np.ones(use.shape, dtype=np.int32)
        full_mixcomps[use] = mixcomps
    full_mixcomps = mixcomps

    bkgs = [feats[full_mixcomps == k].mean(axis=0) for k in xrange(K)]
    return np.asarray([(full_mixcomps == k) & use for k in xrange(K)]), bkgs

def _partition_bkg_files_star(args):
    return _partition_bkg_files(*args)

def _create_kernel_for_mixcomp(mixcomp, settings, bb, indices, files, neg_files):
    im_size = settings['detector']['image_size']
    size = gv.bb.size(bb)
    orig_size = size
    
    gen = generate_random_patches(neg_files, size, seed=0)
    descriptor = gv.load_descriptor(settings)

    eps = settings['detector']['min_probability']
    radii = settings['detector']['spread_radii']
    psize = settings['detector']['subsample_size']
    duplicates = settings['detector'].get('duplicates', 1)
    cb = settings['detector'].get('crop_border')

    bkg = None
    kern = None
    total = 0

    alpha_cum = None


    setts = dict(spread_radii=radii, subsample_size=psize, crop_border=cb)

    for index in indices: 
        ag.info("Processing image of index {0} and mixture component {1}".format(index, mixcomp))
        gray_im, alpha = _load_cad_image(files[index], im_size, bb)

        bin_alpha = alpha > 0.05

        if alpha_cum is None:
            alpha_cum = bin_alpha.astype(np.uint32)
        else:
            alpha_cum += bin_alpha 

        for dup in xrange(duplicates):
            neg_im = gen.next()
            superimposed_im = neg_im * (1 - alpha) + gray_im * alpha

            bkg_feats = descriptor.extract_features(neg_im, settings=setts)
            #bkg_feats = gv.sub.subsample(bkg_feats, psize)
        
            if bkg is None:
                bkg = bkg_feats.astype(np.uint32)
            else:
                bkg += bkg_feats

            feats = descriptor.extract_features(superimposed_im, settings=setts)
            #feats = gv.sub.subsample(feats, psize)

            if kern is None:
                kern = feats.astype(np.uint32)
            else:
                kern += feats

            total += 1
    
    kern = kern.astype(np.float64) / total 
    #kern = np.clip(kern, eps, 1-eps)

    bkg = bkg.astype(np.float64) / total

    support = alpha_cum.astype(np.float64) / len(indices)

    #kernels.append(kern)
    return kern, bkg, orig_size, support 

def _create_kernel_for_mixcomp_star(args):
    return _create_kernel_for_mixcomp(*args)

def _create_kernel_for_mixcomp2(mixcomp, settings, bb, indices, files, neg_files, neg_selectors=None):
    #return 0, 1, 2, 3 
        
    im_size = settings['detector']['image_size']
    size = gv.bb.size(bb)
    orig_size = size
    
    gen = generate_random_patches(neg_files, size, seed=0)
    if neg_selectors is not None:
        gen = itertools.compress(gen, neg_selectors)
    gen = itertools.cycle(gen)
    #gen = generate_random_patches(neg_files, size, seed=mixcomp)
    
    descriptor = gv.load_descriptor(settings)

    eps = settings['detector']['min_probability']
    radii = settings['detector']['spread_radii']
    psize = settings['detector']['subsample_size']
    duplicates = settings['detector'].get('duplicates', 1)
    cb = settings['detector'].get('crop_border')

    bkg = None
    kern = None
    total = 0

    alpha_cum = None

    setts = dict(spread_radii=radii, subsample_size=psize, crop_border=cb)

    for index in indices: 
        ag.info("Processing image of index {0} and mixture component {1}".format(index, mixcomp))
        gray_im, alpha = _load_cad_image(files[index], im_size, bb)

        bin_alpha = alpha > 0.05

        if alpha_cum is None:
            alpha_cum = bin_alpha.astype(np.uint32)
        else:
            alpha_cum += bin_alpha 

        for dup in xrange(duplicates):
            neg_im = gen.next()
            superimposed_im = neg_im * (1 - alpha) + gray_im * alpha

            bkg_feats = descriptor.extract_features(neg_im, settings=setts)
            #bkg_feats = gv.sub.subsample(bkg_feats, psize)
        
            if bkg is None:
                bkg = bkg_feats.astype(np.uint32)
            else:
                bkg += bkg_feats

            feats = descriptor.extract_features(superimposed_im, settings=setts)
            #feats = gv.sub.subsample(feats, psize)

            if kern is None:
                kern = feats.astype(np.uint32)
            else:
                kern += feats

            total += 1
    
    kern = kern.astype(np.float64) / total 
    #kern = np.clip(kern, eps, 1-eps)

    bkg = bkg.astype(np.float64) / total

    support = alpha_cum.astype(np.float64) / len(indices)

    #kernels.append(kern)
    return kern, bkg, orig_size, support 


def _create_kernel_for_mixcomp2_star(args):
    return _create_kernel_for_mixcomp2(*args)

def _classify__(neg_feats, pos_feats, bkgs):
    K = len(bkgs)
    collapsed_feats = np.apply_over_axes(np.mean, neg_feats, [0, 1])
    scores = [-np.sum((collapsed_feats - bkgs[k])**2) for k in xrange(K)]
    bkg_id = np.argmax(scores)
    return bkg_id

def logf(B, bmf, L):
    return -(B - bmf)**2 / (2 * 1 / L * bmf * (1 - bmf)) - 0.5 * np.log(1 / L * bmf * (1 - bmf))

def _classify(neg_feats, pos_feats, bkgs):
    K = len(bkgs)   
    collapsed_feats = np.apply_over_axes(np.mean, neg_feats, [0, 1])
    scores = [logf(collapsed_feats, np.clip(bkgs[k], 0.01, 0.99), 10).sum() for k in xrange(K)]
    bkg_id = np.argmax(scores)
    return bkg_id
    
def _create_kernel_for_mixcomp3(mixcomp, settings, bb, indices, files, neg_files, bkgs):
    #return 0, 1, 2, 3 
    K = len(bkgs)
        
    im_size = settings['detector']['image_size']
    size = gv.bb.size(bb)
    orig_size = size
    
    gen = generate_random_patches(neg_files, size, seed=0)
    
    descriptor = gv.load_descriptor(settings)

    eps = settings['detector']['min_probability']
    radii = settings['detector']['spread_radii']
    psize = settings['detector']['subsample_size']
    duplicates = settings['detector'].get('duplicates', 1)
    cb = settings['detector'].get('crop_border')

    all_kern = [None for k in xrange(K)]
    all_bkg = [None for k in xrange(K)]
    totals = np.zeros(K) 

    alpha_cum = None

    setts = dict(spread_radii=radii, subsample_size=psize, crop_border=cb)

    for index in indices: 
        ag.info("Processing image of index {0} and mixture component {1}".format(index, mixcomp))
        gray_im, alpha = _load_cad_image(files[index], im_size, bb)

        bin_alpha = alpha > 0.05

        if alpha_cum is None:
            alpha_cum = bin_alpha.astype(np.uint32)
        else:
            alpha_cum += bin_alpha 

        for dup in xrange(duplicates):
            neg_im = gen.next()
            neg_feats = descriptor.extract_features(neg_im, settings=setts)
            superimposed_im = neg_im * (1 - alpha) + gray_im * alpha
            feats = descriptor.extract_features(superimposed_im, settings=setts)

            bkg_id = _classify(neg_feats, feats, bkgs)

            #bkg_feats = gv.sub.subsample(bkg_feats, psize)
        
            if all_bkg[bkg_id] is None:
                all_bkg[bkg_id] = neg_feats.astype(np.uint32)
            else:
                all_bkg[bkg_id] += neg_feats

            #feats = gv.sub.subsample(feats, psize)

            if all_kern[bkg_id] is None:
                all_kern[bkg_id] = feats.astype(np.uint32)
            else:
                all_kern[bkg_id] += feats

            totals[bkg_id] += 1
    
    all_kern = [all_kern[k].astype(np.float64) / totals[k] for k in xrange(K)]
    all_bkg = [all_bkg[k].astype(np.float64) / totals[k] for k in xrange(K)]
    
    #kern = kern.astype(np.float64) / total 
    #kern = np.clip(kern, eps, 1-eps)

    #bkg = bkg.astype(np.float64) / total

    support = alpha_cum.astype(np.float64) / len(indices)

    #kernels.append(kern)
    return all_kern, all_bkg, orig_size, support 


def _create_kernel_for_mixcomp3_star(args):
    return _create_kernel_for_mixcomp3(*args)


def _load_cad_image(fn, im_size, bb):
    im = gv.img.load_image(fn)
    im = gv.img.resize(im, im_size)
    im = gv.img.crop_to_bounding_box(im, bb)
    gray_im, alpha = gv.img.asgray(im), im[...,3] 
    return gray_im, alpha
        
def _calc_standardization_for_mixcomp(mixcomp, settings, bb, kern, bkg, indices, files, neg_files):
    im_size = settings['detector']['image_size']
    size = gv.bb.size(bb)

    # Use the same seed for all mixture components! That will make them easier to compare,
    # without having to sample to infinity.
    gen = generate_random_patches(neg_files, size, seed=0)
    descriptor = gv.load_descriptor(settings)

    eps = settings['detector']['min_probability']
    radii = settings['detector']['spread_radii']
    psize = settings['detector']['subsample_size']
    duplicates = settings['detector'].get('duplicates', 1) 
    cb = settings['detector'].get('crop_border')

    total = 0

    neg_llhs = []
    llhs = []

    kern = np.clip(kern, eps, 1 - eps)
    bkg = np.clip(bkg, eps, 1 - eps)
    weights = np.log(kern / (1 - kern) * ((1 - bkg) / bkg))

    for index in indices: 
        ag.info("Standardizing image of index {0} and mixture component {1}".format(index, mixcomp))
        gray_im, alpha = _load_cad_image(files[index], im_size, bb)
        for dup in xrange(duplicates):
            neg_im = gen.next()
            superimposed_im = neg_im * (1 - alpha) + gray_im * alpha
            #superimposed_im = neg_im

            neg_feats = descriptor.extract_features(neg_im, settings=dict(spread_radii=radii, subsample_size=psize, crop_border=cb))
            neg_llh = float((weights * neg_feats).sum())
            neg_llhs.append(neg_llh)
            #if neg_llh == 0:
                #import pdb; pdb.set_trace()

            feats = descriptor.extract_features(superimposed_im, settings=dict(spread_radii=radii, subsample_size=psize, crop_border=cb))
            #feats = gv.sub.subsample(feats, psize)

            llh = float((weights * feats).sum())
            llhs.append(llh)

    #np.save('llhs-{0}.npy'.format(mixcomp), llhs)

    return np.asarray(llhs), np.asarray(neg_llhs)
    #return np.mean(llhs), np.std(llhs)

def _calc_standardization_for_mixcomp_star(args):
    return _calc_standardization_for_mixcomp(*args)

def _calc_standardization_for_mixcomp2(mixcomp, settings, bb, kern, bkg, indices, files, neg_files, neg_selectors=None, duplicates_mult=1):
    im_size = settings['detector']['image_size']
    size = gv.bb.size(bb)

    # Use the same seed for all mixture components! That will make them easier to compare,
    # without having to sample to infinity.
    gen = generate_random_patches(neg_files, size, seed=0)
    if neg_selectors is not None:
        gen = itertools.compress(gen, neg_selectors)
    gen = itertools.cycle(gen)
    descriptor = gv.load_descriptor(settings)

    eps = settings['detector']['min_probability']
    radii = settings['detector']['spread_radii']
    psize = settings['detector']['subsample_size']
    duplicates = settings['detector'].get('duplicates', 1) * duplicates_mult
    cb = settings['detector'].get('crop_border')

    total = 0

    neg_llhs = []
    llhs = []

    kern = np.clip(kern, eps, 1 - eps)
    bkg = np.clip(bkg, eps, 1 - eps)
    weights = np.log(kern / (1 - kern) * ((1 - bkg) / bkg))
    print indices

    for index in indices: 
        ag.info("Standardizing image of index {0} and mixture component {1}".format(index, mixcomp))
        gray_im, alpha = _load_cad_image(files[index], im_size, bb)
        for dup in xrange(duplicates):
            neg_im = gen.next()
            superimposed_im = neg_im * (1 - alpha) + gray_im * alpha
            #superimposed_im = neg_im

            neg_feats = descriptor.extract_features(neg_im, settings=dict(spread_radii=radii, subsample_size=psize, crop_border=cb))
            neg_llh = float((weights * neg_feats).sum())
            neg_llhs.append(neg_llh)

            feats = descriptor.extract_features(superimposed_im, settings=dict(spread_radii=radii, subsample_size=psize, crop_border=cb))
            llh = float((weights * feats).sum())
            llhs.append(llh)

    #np.save('llhs-{0}.npy'.format(mixcomp), llhs)

    #return np.mean(llhs), np.std(llhs)
    return np.asarray(llhs), np.asarray(neg_llhs)

def _calc_standardization_for_mixcomp2_star(args):
    return _calc_standardization_for_mixcomp2(*args)

def _calc_standardization_for_mixcomp3(mixcomp, settings, bb, all_kern, all_bkg, bkgs, indices, files, neg_files, duplicates_mult=1):
    im_size = settings['detector']['image_size']
    size = gv.bb.size(bb)

    # Use the same seed for all mixture components! That will make them easier to compare,
    # without having to sample to infinity.
    gen = generate_random_patches(neg_files, size, seed=0)
    descriptor = gv.load_descriptor(settings)

    eps = settings['detector']['min_probability']
    radii = settings['detector']['spread_radii']
    psize = settings['detector']['subsample_size']
    duplicates = settings['detector'].get('duplicates', 1) * duplicates_mult
    cb = settings['detector'].get('crop_border')

    new_bkg = np.asarray([np.apply_over_axes(np.mean, bkg, [0, 1]) for bkg in all_bkg])
    pos_bbkgs = np.asarray([np.clip(np.apply_over_axes(np.mean, bkg, [0, 1]), eps, 1-eps) for bkg in all_bkg])
    neg_bbkgs = np.asarray([1 - pos_bkg for pos_bkg in pos_bbkgs])

    total = 0

    all_neg_llhs = [[] for k in xrange(K)]
    all_pos_llhs = [[] for k in xrange(K)]

    all_clipped_kern = [np.clip(kern, eps, 1 - eps) for kern in all_kern] 
    all_clipped_bkg = [np.clip(bkg, eps, 1 - eps) for bkg in all_bkg]
    weights = [np.log(all_clipped_kern[k] / (1 - all_clipped_kern[k]) * ((1 - all_clipped_bkg[k]) / all_clipped_bkg[k])) for k in xrange(K)]
    print indices

    for index in indices: 
        ag.info("Standardizing image of index {0} and mixture component {1}".format(index, mixcomp))
        gray_im, alpha = _load_cad_image(files[index], im_size, bb)
        for dup in xrange(duplicates):
            neg_im = gen.next()
            # Check which component this one is
            neg_feats = descriptor.extract_features(neg_im, settings=dict(spread_radii=radii, subsample_size=psize, crop_border=cb))
            superimposed_im = neg_im * (1 - alpha) + gray_im * alpha
            feats = descriptor.extract_features(superimposed_im, settings=dict(spread_radii=radii, subsample_size=psize, crop_border=cb))

            bkg_id = _classify(neg_feats, feats, bkgs)

            neg_llh = float((weights[bkg_id] * neg_feats).sum())
            all_neg_llhs[bkg_id].append(neg_llh)

            pos_llh = float((weights[bkg_id] * feats).sum())
            all_pos_llhs[bkg_id].append(pos_llh)

    #np.save('llhs-{0}.npy'.format(mixcomp), llhs)

    standardization_info = []

    for k in xrange(K):
        neg_llhs = np.asarray(all_neg_llhs[k])
        pos_llhs = np.asarray(all_pos_llhs[k])
        if len(neg_llhs) > 0 and len(pos_llhs):
            info = _standardization_info_for_linearized_non_parameteric(neg_llhs, pos_llhs)
        else:
            print "FAILED", k
            info = {}
        # Optionally add original likelihoods for inspection
        info['pos_llhs'] = pos_llhs
        info['neg_llhs'] = neg_llhs 
        standardization_info.append(info)

    return standardization_info 

def _calc_standardization_for_mixcomp3_star(args):
    return _calc_standardization_for_mixcomp3(*args)


def _logpdf(x, loc=0.0, scale=1.0):
    return -(x - loc)**2 / (2*scale**2) - 0.5 * np.log(2*np.pi) - np.log(scale)

def _standardization_info_for_linearized_non_parameteric_OLD(neg_llhs, pos_llhs):
    info = {}

    mn = min(np.min(neg_llhs), np.min(pos_llhs))
    mx = max(np.max(neg_llhs), np.max(pos_llhs))
    span = (mn, mx)
    
    # Points along our linearization
    x = np.linspace(span[0], span[1], 100)
    delta = x[1] - x[0]

    neg_logs = np.zeros_like(neg_llhs)
    pos_logs = np.zeros_like(pos_llhs)

    print len(neg_logs), len(pos_logs)

    def score(R, neg_llhs, pos_llhs):
        for j, llh in enumerate(neg_llhs):
            neg_logs[j] = _logpdf(R, loc=llh, scale=200) - np.log(len(neg_logs))

        for j, llh in enumerate(pos_llhs):
            pos_logs[j] = _logpdf(R, loc=llh, scale=200) - np.log(len(pos_logs))

        from scipy.misc import logsumexp
        return logsumexp(pos_logs) - logsumexp(neg_logs)


    y = np.zeros_like(x)
    for i in xrange(len(x)):
        y[i] = score(x[i], neg_llhs, pos_llhs)

    info['start'] = mn
    info['step'] = delta
    info['points'] = y
    return info

def _standardization_info_for_linearized_non_parameteric(neg_llhs, pos_llhs):
    info = {}

    from scipy.stats import norm
    def st(x):
        center = np.mean([np.mean(neg_llhs), np.mean(pos_llhs)])
        #return (norm.ppf(norm.sf(neg_llhs, loc=x, scale=50).mean()) + norm.ppf(norm.sf(pos_llhs, loc=x, scale=50).mean()))/2
        n = -3.0 + norm.ppf(norm.sf(neg_llhs, loc=x, scale=200).mean())
        p = +3.0 + norm.ppf(norm.sf(pos_llhs, loc=x, scale=200).mean())
        if np.fabs(x - center) <= 500:# > center:
            alpha = ((x - center) + 500) / 1000
            return p * alpha + n * (1 - alpha)
        elif x > center + 500:
            return p
        else:
            return n

    mn = min(np.min(neg_llhs), np.min(pos_llhs))
    mx = max(np.max(neg_llhs), np.max(pos_llhs))
    span = (mn, mx)

    x = np.linspace(span[0], span[1], 100)

    y = np.asarray([st(xi) for xi in x])
    #import pdb; pdb.set_trace()
    def finitemax(x):
        return x[np.isfinite(x)].max()
    y = np.r_[y[0], np.asarray([(y[j] if (y[j] >= finitemax(y[:j]) and np.isfinite(y[j])) else finitemax(y[:j]))+0.0001*j for j in xrange(1, len(y))])]
    
    info['start'] = mn
    info['step'] = delta
    info['points'] = y
    return info


def superimposed_model(settings, threading=True):
    offset = settings['detector'].get('train_offset', 0)
    limit = settings['detector'].get('train_limit')
    num_mixtures = settings['detector']['num_mixtures']
    assert limit is not None, "Must specify limit in the settings file"
    files = sorted(glob.glob(settings['detector']['train_dir']))[offset:offset+limit]
    neg_files = sorted(glob.glob(settings['detector']['neg_dir']))

    # Train a mixture model to get a clustering of the angles of the object
    descriptor = gv.load_descriptor(settings)
    detector = gv.BernoulliDetector(num_mixtures, descriptor, settings['detector'])

    print "Checkpoint 1"

    bkg_type = detector.settings['bkg_type']
    testing_type = detector.settings['testing_type']
    detector.settings['bkg_type'] = None
    detector.settings['testing_type'] = None

    detector.train_from_images(files)

    detector.settings['bkg_type'] = bkg_type
    detector.settings['testing_type'] = testing_type

    print "Checkpoint 2"

    comps = detector.mixture.mixture_components()
    each_mix_N = np.bincount(comps, minlength=num_mixtures)

    print "Checkpoint 3"

    #for fn in glob.glob('toutputs/*.png'):
        #os.remove(fn)

    if 0:
        from shutil import copyfile
        for mixcomp in xrange(detector.num_mixtures):
            indices = np.where(comps == mixcomp)[0]
            for i in indices:
                copyfile(files[i], 'toutputs/mixcomp-{0}-index-{1}.png'.format(mixcomp, i))

    print "Checkpoint 4"

    support = detector.support 

    kernels = []

    #print "TODO, quitting"
    #return detector

    psize = settings['detector']['subsample_size']

    def get_full_size_bb(k):
        bb = detector.bounding_box_for_mix_comp(k)
        return tuple(bb[i] * psize[i%2] for i in xrange(4))

    def iround(x):
        return int(round(x))

    def make_bb(bb, max_bb):
        # First, make it integral
        bb = (iround(bb[0]), iround(bb[1]), iround(bb[2]), iround(bb[3]))
        bb = gv.bb.inflate(bb, 4)
        bb = gv.bb.intersection(bb, max_bb)
        return bb

    print "Checkpoint 5"

    max_bb = (0, 0) + detector.settings['image_size']
    bbs = [make_bb(get_full_size_bb(k), max_bb) for k in xrange(detector.num_mixtures)]

    print "Checkpoint 6"

    #for mixcomp in xrange(num_mixtures):
    
    if threading:
        from multiprocessing import Pool
        p = Pool(7)
        # Order is important, so we can't use imap_unordered
        imapf = p.imap
    else:
        #from itertools import imap as imapf
        imapf = itertools.imap


    

    argses = [(i, settings, bbs[i], list(np.where(comps == i)[0]), files, neg_files) for i in xrange(detector.num_mixtures)] 

    print "Checkpoint 7"

    #all_mixcomps = [_partition_bkg_files() for 
    
    #argses = [(i,) for i in xrange(detector.num_mixtures)] 
    kernels = []
    bkgs = []
    bkg_centers = []
    orig_sizes = []
    new_support = []
    neg_selectors = [None] * 100

    ONE_MIXCOMP = None


    if ONE_MIXCOMP is not None:
        kern, bkg, orig_size, sup = _create_kernel_for_mixcomp_star(argses[ONE_MIXCOMP]) 
        kernels.append(kern)
        bkgs.append(bkg)
        orig_sizes.append(orig_size)
        new_support.append(sup)
        detector.num_mixtures = 1

        detector.settings['per_mixcomp_bkg'] = True
    
    else:
        print "Checkpoint 8"
        all_negs = []
        if 1:
            neg_selectors = []
            argses2 = [(i, len(np.where(comps == i)[0]) * settings['detector']['duplicates'], settings, gv.bb.size(bbs[i]), neg_files, files, bbs[0]) for i in xrange(detector.num_mixtures)]
            for m, (neg_sel, bkg_center) in enumerate(imapf(_partition_bkg_files_star, argses2)):
                #print "neg_ims size:", map(np.shape, neg_ims)
                #all_negs.append(neg_ims)
                for k, negs in enumerate(neg_sel):
                    print "COMP", k, negs.sum()
                neg_selectors.append(neg_sel)
                bkg_centers.extend(bkg_center)

        elif 1:
            # TODO: Temporary
            #brick_files = sorted(glob.glob(os.path.expanduser('/var/tmp/d/bricks/*')))
            N1 = len(neg_files)
            N2 = len(brick_files)
            #neg_files = 
            neg_files = neg_files + brick_files
            neg_selectors = [[
                [1] * N1 + [0] * N2,
                [0] * N1 + [1] * N2,
            ]]

            if 0:
                Ns = [len(np.where(comps == k)[0]) * settings['detector']['duplicates'] for k in xrange(detector.num_mixtures)]

                for i, N in enumerate(Ns):
                    size = gv.bb.size(bbs[i])
                    neg_ims = []
                    for k in xrange(K): 
                        if k == 0:
                            gen = generate_random_patches(neg_files, size, seed=i)
                            ims = list(itertools.islice(gen, N))
                        else:
                            gen = generate_random_patches(brick_files, size, seed=i)
                            ims = list(itertools.islice(gen, N))
                        neg_ims.append(ims)
                    all_negs.append(neg_ims)
        else:
            kernels = []
            bkgs = []
            Ns = [len(np.where(comps == k)[0]) * settings['detector']['duplicates'] for k in xrange(detector.num_mixtures)]

            argses = [(i, settings, bbs[i], list(np.where(comps == i)[0]), files, neg_files) for i in xrange(detector.num_mixtures)] 

            for kern, bkg, orig_size, sup in imapf(_create_kernel_for_mixcomp2_star, argses):
                kernels.append(kern)

                if detector.settings.get('collapse_bkg'):
                    bkg = np.apply_over_axes(np.mean, bkg, [0, 1]).ravel()
                bkgs.append(bkg)
                orig_sizes.append(orig_size)
                new_support.append(sup)


            #argses = [(i, settings, bbs[i], kernels[i], bkgs[i], list(np.where(comps == i)[0]), files, neg_files) for i in xrange(detector.num_mixtures)]
            argses = [(i, settings, bbs[i], kernels[i], bkgs[i], list(np.where(comps == i)[0]), files, neg_files, None, 5) for i in xrange(detector.num_mixtures)]
            #for i, (mean, std) in enumerate(imapf(_calc_standardization_for_mixcomp2_star, argses)):
            neg_selectors = []
            for i, (llhs, neg_llhs) in enumerate(imapf(_calc_standardization_for_mixcomp2_star, argses)):
                # Extract the backgrounds scoring the most as the object 
                from scipy.stats import mstats
                quantiles = mstats.mquantiles(neg_llhs, np.linspace(0, 1, 10)) 
                mn, mx = quantiles[-2:]
                avg = np.mean([mn, mx])
                dist = mx - avg
                selectors = (np.fabs(neg_llhs - avg) <= dist)
                N = selectors.sum()
                #ims = [neg_ims[i][j % N] for j in indices]
                neg_selectors.append(selectors) 
                #print 'IMAGES', len(ims)
                #all_negs.append([ims])

        if 0:
            import pylab as plt
            for comp, neg_ims_list in enumerate(all_negs):
                for bkg_comp, neg_ims in enumerate(neg_ims_list): 
                    for i, im in enumerate(neg_ims):
                        gv.img.save_image(im, 'bkg-mixtures/comp{0}-bkg{1}-{2}.png'.format(comp, bkg_comp, i))
            


        print "Checkpoint 9"

        if 0:
            argses = [(i, settings, bbs[i], list(np.where(comps == i)[0]), files, neg_files, neg_selectors[i][k]) for i in xrange(detector.num_mixtures) for k in xrange(K)] 
            for kern, bkg, orig_size, sup in imapf(_create_kernel_for_mixcomp2_star, argses):
                print "HERE"
                kernels.append(kern)

                if detector.settings.get('collapse_bkg'):
                    bkg = np.apply_over_axes(np.mean, bkg, [0, 1]).ravel()
                bkgs.append(bkg)
                orig_sizes.append(orig_size)
                new_support.append(sup)
        else:
            # FOR ONLY ONE MIXCOMP
            all_kern, all_bkg, orig_size, sup = _create_kernel_for_mixcomp3(0, settings, bbs[0], np.where(comps == 0)[0], files, neg_files, bkg_centers)
            for k in xrange(K):
                kernels.append(all_kern[k]) 
                bkgs.append(all_bkg[k])
                orig_sizes.append(orig_size)
                new_support.append(sup)
                

        print "Checkpoint 10"

        detector.settings['per_mixcomp_bkg'] = True

    # TODO
    detector.num_mixtures *= K

    detector.kernel_templates = kernels
    detector.kernel_sizes = orig_sizes
    detector.settings['kernel_ready'] = True
    detector.use_alpha = False
    detector.support = new_support

    # Determine the background
    ag.info("Determining background")
    #spread_bkg = np.mean([kern[:2].reshape((-1, kern.shape[-1])).mean(axis=0) for kern in kernels], axis=0)
    #spread_bkg = np.mean([kern.reshape((-1, kern.shape[-1])).mean(axis=0) for kern in kernels], axis=0)
    #spread_bkg = kernels[0][1].mean(axis=0)
    if ONE_MIXCOMP is not None:
        eps = detector.settings['min_probability']
        detector.fixed_bkg = None
        detector.fixed_spread_bkg = bkgs
    else:
        #spread_bkg = fetch_bkg_model(settings, neg_files)

        #eps = detector.settings['min_probability']
        #spread_bkg = np.clip(spread_bkg, eps, 1 - eps)

        #print 'spread_bkg shape:', spread_bkg.shape
        detector.fixed_bkg = None # Not needed, since kernel_ready is True
        #detector.fixed_spread_bkg = spread_bkg
        detector.fixed_spread_bkg = bkgs

    detector.settings['bkg_type'] = 'from-file'

    # Determine the standardization values
    ag.info("Determining standardization values")

    #fixed_train_mean = np.zeros(detector.num_mixtures)
    #detector.fixed_train_mean = []
    #fixed_train_std = np.ones(detector.num_mixtures)

    if testing_type in ('fixed', 'non-parametric'):
        detector.standardization_info = []
        if ONE_MIXCOMP is None:

            detector.standardization_info = []
            if testing_type == 'non-parametric':
                argses = [(i, settings, bbs[i], kernels[i*K:i*K + K], bkgs[i*K:i*K + K], bkg_centers, list(np.where(comps == i)[0]), files, neg_files, 100) for i in xrange(detector.num_mixtures//K)]
                for i, si in enumerate(imapf(_calc_standardization_for_mixcomp3_star, argses)):
                    detector.standardization_info += si 

            else:
                #argses = [(i, settings, bbs[i], kernels[i], bkgs[i], list(np.where(comps == i)[0]), files, neg_files) for i in xrange(detector.num_mixtures)]
                argses = [(i, settings, bbs[i//K], kernels[i], bkgs[i], list(np.where(comps == i//K)[0]), files, neg_files, neg_selectors[i//K][i%K]) for i in xrange(detector.num_mixtures)]
                print "STOP 2"
                #for i, (mean, std) in enumerate(imapf(_calc_standardization_for_mixcomp2_star, argses)):
                for i, (llhs, neg_llhs) in enumerate(imapf(_calc_standardization_for_mixcomp2_star, argses)):
                    #detector.standardization_info.append(dict(mean=mean, std=std))
                    #detector.fixed_train_mean.append({
                    #    'pos_llhs': llhs,
                    #    'neg_llhs': neg_llhs,
                    #})
                    print "STOP 3"

                    if testing_type == 'fixed':
                        mean = np.mean(neg_llhs)
                        std = np.std(neg_llhs)
                        detector.standardization_info.append(dict(mean=mean, std=std))
                    elif testing_type == 'non-parametric':
                        info = _standardization_info_for_linearized_non_parameteric(neg_llhs, llhs)
                        # Optionally add original likelihoods for inspection
                        info['pos_llhs'] = llhs
                        info['neg_llhs'] = neg_llhs 
                        detector.standardization_info.append(info)

                argses = [(i, settings, bbs[i//K], bkgs[i], 0.5 * np.ones(bkgs[i].shape), list(np.where(comps == i//K)[0]), files, neg_files, neg_selectors[i//K][i%K]) for i in xrange(detector.num_mixtures)]
                for i, (llhs, neg_llhs) in enumerate(imapf(_calc_standardization_for_mixcomp2_star, argses)):

                    detector.standardization_info[i]['bkg_mean'] = np.mean(neg_llhs)
                    detector.standardization_info[i]['bkg_std'] = np.std(neg_llhs)

    detector.settings['testing_type'] = testing_type 
    #detector.settings['testing_type'] = 'NEW'

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
