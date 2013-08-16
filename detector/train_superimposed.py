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
import itertools
from collections import namedtuple
from superimpose_experiment import generate_random_patches

#KMEANS = False 
#LOGRATIO = True 
KMEANS = True
LOGRATIO = True 

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
    

def _partition_bkg_files(count, settings, size, neg_files, files, bb, num_bkg_mixtures):
    im_size = settings['detector']['image_size'] # TEMP

    gen = generate_random_patches(neg_files, size, seed=0)

    descriptor = gv.load_descriptor(settings)

    radii = settings['detector']['spread_radii']
    psize = settings['detector']['subsample_size']
    cb = settings['detector'].get('crop_border')
    setts = dict(spread_radii=radii, subsample_size=psize, crop_border=cb)

    #np.apply_over_axes(np.mean, descriptor.parts, [1, 2]).reshape((-1, 4))
    
    orrs = np.apply_over_axes(np.mean, descriptor.parts, [1, 2]).reshape((-1, 4))
    norrs = orrs / np.expand_dims(orrs.sum(axis=1), 1)

    if 0:
        bkgs = np.zeros((num_bkg_mixtures, descriptor.num_features))
        chunk = descriptor.num_features // num_bkg_mixtures
        for m in xrange(num_bkg_mixtures):
            fr = m * chunk
            to = min((m + 1) * chunk, descriptor.num_features)
            bkgs[m,fr:to] = 0.6 
        return bkgs
       
    neg_ims = []
    feats = []
    use = np.ones(count, dtype=bool)

    #prnd = np.random.RandomState(1)
    # TEMP
    #return [], np.clip(prnd.normal(loc=0.05, scale=0.05, size=(K, descriptor.num_features)), 0.0, 1.0)

    means = []
    stds = []

    orientations = []

    for i, neg_im in enumerate(gen):
        if i == count:
            break

        gray_im, alpha = _load_cad_image(files[i%len(files)], im_size, bb)
        superimposed_im = neg_im * (1 - alpha) + gray_im * alpha

        # TODO: USING WITH THE CAR NOW!
        im = superimposed_im
        #im = neg_im

        if i % 100 == 0:
            ag.info("Loading bkg im {0}".format(i))
    
        all_feat = descriptor.extract_features(im, settings=setts)

        #feat = np.apply_over_axes(np.mean, all_feat, [0, 1]).ravel() 
        #feat = np.apply_over_axes(np.sum, all_feat, [0, 1]).ravel() - 
        inner_padding = -2
        area = np.prod(all_feat.shape[:2]) - max(all_feat.shape[0] + 2*inner_padding, 0) * max(all_feat.shape[1] + 2*inner_padding, 0)

        feat = np.apply_over_axes(np.sum, all_feat, [0, 1]).ravel() - np.apply_over_axes(np.sum, all_feat[-inner_padding:inner_padding, -inner_padding:inner_padding], [0, 1]).ravel()
        feat = feat.astype(np.float64) / area
    
        if 0:
            if 0.10 < feat.mean() < 0.20 and \
               0.10 < feat.std() < 0.15:
                feats.append(feat)
            else:
                use[i] = False
        feats.append(feat)
        
        means.append(feat.mean()) 
        stds.append(feat.std())

        #angles = np.hstack([np.cos(descriptor.orientations) * feat, np.sin(descriptor.orientations) * feat])
        #orientations.append(angles)


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

        #neg_ims.append(neg_im)

    feats = np.asarray(feats)
    orientations = np.asarray(orientations)

    means = np.asarray(means)
    stds = np.asarray(stds)

    #np.savez('means.npz', means=means, stds=stds)
    #neg_ims = np.asarray(neg_ims)

    print "Partitioning..."
    if 0:
        from sklearn.cluster import KMeans
        if num_bkg_mixtures > 1:
            mixture = GMM(num_bkg_mixtures) 
            #mixture = KMeans(num_bkg_mixtures)
            mixture.fit(orientations)
            #mixcomps = mixture.predict(orientations)
            mixcomps = mixture.labels_
        else:
            mixcomps = np.zeros(len(feats))

        if 1:
            full_mixcomps = np.ones(use.shape, dtype=np.int32)
            full_mixcomps[use] = mixcomps
        else:
            full_mixcomps = mixcomps
        
        bkgs = np.asarray([feats[full_mixcomps == k].mean(axis=0) for k in xrange(num_bkg_mixtures)])
        return bkgs 
    elif KMEANS:
        if num_bkg_mixtures > 1:
            from sklearn.cluster import KMeans
            #from sklearn.mixture import GMM
        
            #mixture = GMM(num_bkg_mixtures)
            mixture = KMeans(num_bkg_mixtures)
            print "Created clustering"
            mixture.fit(feats)
            print "Running clustering"

            #mixcomps = mixture.predict(feats)
            mixcomps = mixture.labels_
        else:
            mixcomps = np.zeros(len(feats))

        print "Predicted"

        #return [neg_ims[mixcomps==i] for i in xrange(K)]
        if 0:
            full_mixcomps = np.ones(use.shape, dtype=np.int32)
            full_mixcomps[use] = mixcomps
        else:
            full_mixcomps = mixcomps

        # TODO: Why not just mixture.cluster_centers_ directly?
        #bkgs = np.asarray([feats[(full_mixcomps == k) & use].mean(axis=0) for k in xrange(num_bkg_mixtures)])
        #bkgs = np.asarray([feats[(full_mixcomps == k) & use].mean(axis=0) for k in xrange(num_bkg_mixtures)])
        bkgs = np.asarray([feats[full_mixcomps == k].mean(axis=0) for k in xrange(num_bkg_mixtures)])
        return bkgs
    else:
        model = gv.BetaMixture(num_bkg_mixtures)
        model.fit(feats)
        
        return model.theta_ 

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

if KMEANS:

    def _scores(feats, mixture_params):
        K = len(mixture_params)
        from gv.fast import bkg_model_dists 
        return -bkg_model_dists(feats, mixture_params, feats.shape[:2], padding=0, inner_padding=-2)[0,0]
    
    def _classify(neg_feats, pos_feats, mixture_params):
        K = len(mixture_params)
        from gv.fast import bkg_model_dists 
        return np.argmax(_scores(pos_feats, mixture_params))

    if 0:
        def _classify(neg_feats, pos_feats, mixture_params):
            K = len(mixture_params)
            # TODO: NEW USING POS HERE!!!!
            collapsed_feats = np.apply_over_axes(np.mean, pos_feats, [0, 1])
            scores = [-np.sum((collapsed_feats - mixture_params[k])**2) for k in xrange(K)]
            bkg_id = np.argmax(scores)
            return bkg_id

else:
    def _classify(neg_feats, pos_feats, mixture_params):
        from scipy.stats import beta
        M = len(mixture_params)
        # TODO: NEW USING POS HERE!!!!
        collapsed_feats = np.apply_over_axes(np.mean, pos_feats, [0, 1]).ravel()
        collapsed_feats = np.clip(collapsed_feats, 0.01, 1-0.01)
        D = collapsed_feats.shape[0]
        
        qlogs = np.zeros(M)
        for m in xrange(M):
            #v = qlogs[m] 
            v = 0.0
            for d in xrange(D):
                v += beta.logpdf(collapsed_feats[d], mixture_params[m,d,0], mixture_params[m,d,1])
            qlogs[m] = v

        bkg_id = qlogs.argmax()
        return bkg_id

if 0:
    def logf(B, bmf, L):
        return -(B - bmf)**2 / (2 * 1 / L * bmf * (1 - bmf)) - 0.5 * np.log(1 / L * bmf * (1 - bmf))

    def _classify(neg_feats, pos_feats, bkgs):
        K = len(bkgs)   
        collapsed_feats = np.apply_over_axes(np.mean, neg_feats, [0, 1])
        scores = [logf(collapsed_feats, np.clip(bkgs[k], 0.01, 0.99), 10).sum() for k in xrange(K)]
        bkg_id = np.argmax(scores)
        return bkg_id
    
def _create_kernel_for_mixcomp3(mixcomp, settings, bb, indices, files, neg_files, bkg_mixture_params):
    #return 0, 1, 2, 3 
    K = len(bkg_mixture_params)
        
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

    all_kern = [None] * K
    all_bkg = [None] * K
    totals = np.zeros(K) 

    alpha_cum = None

    setts = dict(spread_radii=radii, subsample_size=psize, crop_border=cb)
    counts = np.zeros(K)

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

            if K > 1:
                # TODO: THIS IS EXTREMELY TODO, FIXME NEW HELPME!
                #if index == 0 and dup == 0:
                #    bkg_id = 2 
                #else:
                bkg_id = _classify(neg_feats, feats, bkg_mixture_params)
            else:
                bkg_id = 0

            counts[bkg_id] += 1

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

    print 'COUNTS', counts

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

def _calc_standardization_for_mixcomp3(mixcomp, settings, bb, all_kern, all_bkg, bkg_mixture_params, indices, files, neg_files, duplicates_mult=1):
    im_size = settings['detector']['image_size']
    size = gv.bb.size(bb)
    K = len(bkg_mixture_params)

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

            if K > 1:
                scores_neg = _scores(neg_feats, bkg_mixture_params)
                scores_pos = _scores(feats, bkg_mixture_params)

                bkg_id_neg = np.argmax(scores_neg)
                bkg_id_pos = np.argmax(scores_pos)
    
                for bkg_id in np.where(scores_neg > scores_neg.max() - 0.001)[0]:
                    neg_llh = float((weights[bkg_id] * neg_feats).sum())
                    all_neg_llhs[bkg_id].append(neg_llh)

                for bkg_id in np.where(scores_pos > scores_pos.max() - 0.001)[0]:
                    pos_llh = float((weights[bkg_id] * feats).sum())
                    all_pos_llhs[bkg_id].append(pos_llh)
                

                #bkg_id_neg = _classify(None, neg_feats, bkg_mixture_params)
                #bkg_id_pos = _classify(None, feats, bkg_mixture_params)
            else:
                bkg_id_neg = 0
                bkg_id_pos = 0

                neg_llh = float((weights[bkg_id_neg] * neg_feats).sum())
                all_neg_llhs[bkg_id_neg].append(neg_llh)

                pos_llh = float((weights[bkg_id_pos] * feats).sum())
                all_pos_llhs[bkg_id_pos].append(pos_llh)

    #np.save('llhs-{0}.npy'.format(mixcomp), llhs)

    standardization_info = []

    for k in xrange(K):

        neg_llhs = np.asarray(all_neg_llhs[k])
        if 0:
            #neg_llhs = neg_llhs.reshape((40, -1))
            #neg_llhs = neg_llhs.max(axis=0)
            neg_llhs = np.asarray(map(np.max, np.array_split(neg_llhs, neg_llhs.size // 20)))

        pos_llhs = np.asarray(all_pos_llhs[k])
        if 0:
            #pos_llhs = pos_llhs.reshape((5, -1))
            #pos_llhs = pos_llhs.min(axis=0)
            #pos_llhs = pos_llhs[
            pass

        if len(neg_llhs) > 0 and len(pos_llhs) > 0:
            info = _standardization_info_for_linearized_non_parameteric(neg_llhs, pos_llhs)
        else:
            print "FAILED", k
            info = {'start': 0.0, 'step': 1.0, 'points': np.asarray([0.0, 1.0])}
        # Optionally add original likelihoods for inspection
        info['pos_llhs'] = pos_llhs
        info['neg_llhs'] = neg_llhs 
        standardization_info.append(info)

    return standardization_info 

def _calc_standardization_for_mixcomp3_star(args):
    return _calc_standardization_for_mixcomp3(*args)


def _logpdf(x, loc=0.0, scale=1.0):
    #return -(np.clip((x - loc) / scale, -4, 4)*scale)**2 / (2*scale**2) - 0.5 * np.log(2*np.pi) - np.log(scale)
    return -(x - loc)**2 / (2*scale**2) - 0.5 * np.log(2*np.pi) - np.log(scale)

if LOGRATIO:
    def _standardization_info_for_linearized_non_parameteric(neg_llhs, pos_llhs):
        info = {}

        mn = min(np.min(neg_llhs), np.min(pos_llhs))
        mx = max(np.max(neg_llhs), np.max(pos_llhs))
        span = (mn, mx)
        
        # Points along our linearization
        x = np.linspace(span[0], span[1], 100)
        delta = x[1] - x[0]


        #print len(neg_logs), len(pos_logs)

        from  scipy.stats import norm, genextreme

        sig = 250
        N = 50

        def score(R, neg_llhs, pos_llhs):
            neg_logs = np.zeros_like(neg_llhs)
            pos_logs = np.zeros_like(pos_llhs)

            for j, llh in enumerate(neg_llhs):
                #mu_n = norm.ppf(1 - 1/N, loc=llh, scale=sig)
                #sigma_n = norm.ppf(1 - 1/N * np.exp(-1), loc=llh, scale=sig) - mu_n 
                neg_logs[j] = norm.logpdf(R, loc=llh, scale=sig) - np.log(len(neg_logs))
                #neg_logs[j] = genextreme.logpdf(R, 0, loc=mu_n, scale=sigma_n) - np.log(len(neg_logs))

            for j, llh in enumerate(pos_llhs):
                pos_logs[j] = norm.logpdf(R, loc=llh, scale=sig) - np.log(len(pos_logs))

            neg_mean = neg_llhs.mean()
            neg_std = neg_llhs.std()
            pos_mean = pos_llhs.mean()
            pos_std = pos_llhs.std()

            # Combine with regular normals to improve stability
            #neg_logs2 = np.hstack([neg_logs, np.repeat(norm.logpdf(R, loc=neg_mean, scale=neg_std) - np.log(len(neg_logs)), len(neg_logs))])
            #pos_logs2 = np.hstack([pos_logs, np.repeat(norm.logpdf(R, loc=pos_mean, scale=pos_std) - np.log(len(pos_logs)), len(pos_logs))])
            #neg_logs2 = np.repeat(norm.logpdf(R, loc=neg_mean, scale=neg_std) - np.log(len(neg_logs)), len(neg_logs))
            #pos_logs2 = np.repeat(norm.logpdf(R, loc=pos_mean, scale=pos_std) - np.log(len(pos_logs)), len(pos_logs))

            from scipy.misc import logsumexp
            return logsumexp(pos_logs) - logsumexp(neg_logs)
            #return logsumexp(pos_logs2) - logsumexp(neg_logs2)


        y = np.zeros_like(x)
        for i in xrange(len(x)):
            y[i] = score(x[i], neg_llhs, pos_llhs)

        info['start'] = mn
        info['step'] = delta
        info['points'] = y
        return info

else:
    def _standardization_info_for_linearized_non_parameteric(neg_llhs, pos_llhs):
        info = {}

        from scipy.stats import norm
        def st(x):
            ps = np.asarray([np.mean(neg_llhs), np.mean(pos_llhs)])
            center = np.mean(ps)
            #return (norm.ppf(norm.sf(neg_llhs, loc=x, scale=50).mean()) + norm.ppf(norm.sf(pos_llhs, loc=x, scale=50).mean()))/2
            n = -3.0 + norm.ppf(norm.sf(neg_llhs, loc=x, scale=200).mean())
            p = +3.0 + norm.ppf(norm.sf(pos_llhs, loc=x, scale=200).mean())
            diff = np.fabs(ps[0] - ps[1])
            if np.fabs(x - center) <= diff/2:# > center:
                alpha = ((x - center) + diff/2) / diff
                return p * alpha + n * (1 - alpha)
            elif x > center + diff/2:
                return p
            else:
                return n

        def st(x):
            ps = np.asarray([np.mean(neg_llhs), np.mean(pos_llhs)])
            #ps = np.asarray([np.median(neg_llhs), np.median(pos_llhs)])
            k = 6.0 / (ps[1] - ps[0])
            m = 3.0 - k * ps[1]
            return k * x + m

        mn = min(np.min(neg_llhs), np.min(pos_llhs))
        mx = max(np.max(neg_llhs), np.max(pos_llhs))
        span = (mn, mx)

        x = np.linspace(span[0], span[1], 100)
        delta = x[1] - x[0]

        y = np.asarray([st(xi) for xi in x])
        #def finitemax(x):
            #return x[np.isfinite(x)].max()
        #y = np.r_[y[0], np.asarray([(y[j] if (y[j] >= finitemax(y[:j]) and np.isfinite(y[j])) else finitemax(y[:j]))+0.0001*j for j in xrange(1, len(y))])]
        
        info['start'] = mn
        info['step'] = delta
        info['points'] = y
        return info

def _standardization_info_for_linearized_non_parameteric_NONPARAMETRIC(neg_llhs, pos_llhs):
    info = {}

    from scipy.stats import norm
    def st(x, R):
         if x > R.mean():
              return norm.isf(norm.cdf(R, loc=x, scale=150).mean())
         else:
              return norm.ppf(norm.sf(R, loc=x, scale=150).mean())

    def comb(x, sh, nR, pR):
         xn = np.median(nR)
         xp = np.median(pR)
         p = st(x, pR)+sh
         n = st(x, nR)-sh
         diff = xp - xn
         if n > 0 and p < 0:
              #alpha = (x - (xn + xp)/2) / (diff/4)
              #s = 1 / (1 + np.exp(-alpha*7))
              return 0#np.inf#n * (1 - s) + p * s
         elif p > 0:
              return p
         else:
              return n

    mn = min(np.min(neg_llhs), np.min(pos_llhs))
    mx = max(np.max(neg_llhs), np.max(pos_llhs))
    span = (mn, mx)

    x = np.linspace(span[0], span[1], 100)
    delta = x[1] - x[0]

    y = np.asarray([comb(xi, 3, neg_llhs, pos_llhs) for xi in x])
    #def finitemax(x):
        #return x[np.isfinite(x)].max()
    #y = np.r_[y[0], np.asarray([(y[j] if (y[j] >= finitemax(y[:j]) and np.isfinite(y[j])) else finitemax(y[:j]))+0.0001*j for j in xrange(1, len(y))])]
    
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
    neg_files2 = sorted(glob.glob(settings['detector']['neg_dir2']))

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


    

    #argses = [(i, settings, bbs[i], list(np.where(comps == i)[0]), files, neg_files) for i in xrange(detector.num_mixtures)] 

    print "Checkpoint 7"

    kernels = []
    bkgs = []
    bkg_mixture_params = []
    orig_sizes = []
    new_support = []
    neg_selectors = [None] * 100
    im_size = settings['detector']['image_size']

    print "Checkpoint 8"
    all_negs = []

    if 0:
        if detector.num_bkg_mixtures == 1:
            bkg_mixture_params = np.ones(1)
        else:
             
            # TODO: This uses the shape of the first component. This should be fairly arbitrary
            #neglabels = np.zeros(len(neg_files2))
            neg_files_partitions = [[] for k in xrange(detector.num_bkg_mixtures)] 
            for fn in neg_files2:
                # TODO: Very temporary
                s = os.path.basename(fn)[3:] 
                rot = int(s[:s.find('-')])
                neg_files_partitions[rot].append(fn) 
            
            size = gv.bb.size(bbs[0])
            gens = [generate_random_patches(neg_files_partitions[k], size, seed=0) for k in xrange(detector.num_bkg_mixtures)]
            count = len(comps) * settings['detector']['duplicates'] // detector.num_bkg_mixtures

            if KMEANS:
                bkg_mixture_params = np.zeros((detector.num_bkg_mixtures, detector.num_features))
            else:
                bkg_mixture_params = np.zeros((detector.num_bkg_mixtures, detector.num_features, 2))

            radii = settings['detector']['spread_radii']
            psize = settings['detector']['subsample_size']
            cb = settings['detector'].get('crop_border')
            setts = dict(spread_radii=radii, subsample_size=psize, crop_border=cb)

            inner_padding = -2

            for k in xrange(detector.num_bkg_mixtures):
                feats = []
                from itertools import islice
                for i, im in enumerate(islice(gens[k], 0, count)):

                    # TODO: THIS ONLY WORKS FOR ONE MIXCOMP
                    gray_im, alpha = _load_cad_image(files[i % len(files)], im_size, bbs[0])
                    #alpha, cad_im = gv.img.load_image(files[i % len(files)])

                    im = alpha * im + (1 - alpha) * gray_im

                    all_feat = descriptor.extract_features(im, settings=setts)
                    area = np.prod(all_feat.shape[:2]) - max(all_feat.shape[0] + 2*inner_padding, 0) * max(all_feat.shape[1] + 2*inner_padding, 0)

                    feat = np.apply_over_axes(np.sum, all_feat, [0, 1]).ravel() - np.apply_over_axes(np.sum, all_feat[-inner_padding:inner_padding, -inner_padding:inner_padding], [0, 1]).ravel()
                    feat = feat.astype(np.float64) / area
                    feats.append(feat)

                feats = np.asarray(feats)

                if KMEANS:
                    bkg_mixture_params[k] = np.mean(feats, axis=0)
                else:
                    # Run the beta
                    #mixture = gv.BetaMixture(1)
                    ##mixture.fit(feats) 
                    #bkg_mixture_params[k] = mixture.theta_[0]
                    bkg_mixture_params[k] = gv.BetaMixture.fit_beta(feats)
    else:
        bkg_mixture_params = _partition_bkg_files(len(comps) * settings['detector']['duplicates'], settings, gv.bb.size(bbs[0]), neg_files2, files, bbs[0], detector.num_bkg_mixtures)


    #bkg_mixture_params = np.asarray(bkg_mixture_params)
            
    #for fn in neg_files2:

    print "Checkpoint 9"

    # FOR ONLY ONE MIXCOMP
    argses = [(m, settings, bbs[m], np.where(comps == m)[0], files, neg_files, bkg_mixture_params) for m in xrange(detector.num_mixtures)]
    #all_kern, all_bkg, orig_size, sup = _create_kernel_for_mixcomp3
    for all_kern, all_bkg, orig_size, sup in imapf(_create_kernel_for_mixcomp3_star, argses):
        kernels.append(all_kern) 
        bkgs.append(all_bkg)
        orig_sizes.append(orig_size)
        new_support.append(sup)
                
        print "Checkpoint 10"

        detector.settings['per_mixcomp_bkg'] = True

    detector.kernel_templates = kernels
    detector.kernel_sizes = orig_sizes
    detector.settings['kernel_ready'] = True
    detector.use_alpha = False
    detector.support = new_support
    detector.bkg_mixture_params = bkg_mixture_params

    # Determine the background
    ag.info("Determining background")

    detector.fixed_bkg = None
    detector.fixed_spread_bkg = bkgs

    detector.settings['bkg_type'] = 'from-file'

    # Determine the standardization values
    ag.info("Determining standardization values")

    #fixed_train_mean = np.zeros(detector.num_mixtures)
    #detector.fixed_train_mean = []
    #fixed_train_std = np.ones(detector.num_mixtures)

    K = detector.num_bkg_mixtures

    if testing_type in ('fixed', 'non-parametric'):
        detector.standardization_info = []
        if testing_type == 'fixed':
            argses = [(m, settings, bbs[m], kernels[m], bkgs[m], bkg_mixture_params, list(np.where(comps == m)[0]), files, neg_files, 3) for m in xrange(detector.num_mixtures)]
            detector.standardization_info = list(imapf(_calc_standardization_for_mixcomp3_star, argses))
            for all_infos in detector.standardization_info:
                for info in all_infos:
                    info['mean'] = info['neg_llhs'].mean()
                    info['std'] = info['neg_llhs'].std()

        elif testing_type == 'non-parametric':
            #argses = [(m, settings, bbs[m], kernels[m]
            argses = [(m, settings, bbs[m], kernels[m], bkgs[m], bkg_mixture_params, list(np.where(comps == m)[0]), files, neg_files, settings['detector'].get('stand_multiples', 1)) for m in xrange(detector.num_mixtures)]
            detector.standardization_info = list(imapf(_calc_standardization_for_mixcomp3_star, argses))

            for all_infos in detector.standardization_info:
                # Get the mean/std of all negatives after component-wise standardization
                neg_lists = []
                for info in all_infos:
                    neg_R = info['neg_llhs'].copy().reshape((-1, 1))
                    #pos_R = info['pos_llhs'].copy().reshape((-1, 1))
                    from gv.fast import nonparametric_rescore 
                    nonparametric_rescore(neg_R, info['start'], info['step'], info['points'])
                    #nonparametric_rescore(pos_R, dct['start'], dct['step'], dct['points'])
                    neg_lists.append(neg_R.ravel())

                if 0:
                    negs = np.hstack(neg_lists)
                    negs_mean = negs.mean()
                    negs_std = negs.std()
                    
                    # Populate this to all infos
                    for info in all_infos:
                        info['comp_mean'] = negs_mean
                        info['comp_std'] = negs_std
                    
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
