from __future__ import division, print_function, absolute_import

#import argparse

#parser = argparse.ArgumentParser(description='Train mixture model on edge data')
#parser.add_argument('patches', metavar='<patches file>', type=argparse.FileType('rb'), help='Filename of patches file')
#parser.add_argument('model', metavar='<output model file>', type=argparse.FileType('wb'), help='Filename of the output models file')
#parser.add_argument('mixtures', metavar='<number mixtures>', type=int, help='Number of mixture components')
#parser.add_argument('--use-voc', action='store_true', help="Use VOC data to train model")

import gv
import glob
import os
import os.path
import sys
import numpy as np
import amitgroup as ag
import random
import itertools as itr
from copy import deepcopy
from train_superimposed import generate_random_patches, cluster, calc_bbs, get_positives, get_pos_and_neg, arrange_support

ag.set_verbose(True)

PER_IMAGE = 10 

def get_fps(detector, i, fileobj, size):
    ag.info('{0} Initial processing {1}'.format(i, fileobj.img_id))
    gen = generate_random_patches([fileobj.path], size, per_image=PER_IMAGE)
    neg_feats = []

    radii = detector.settings['spread_radii']
    psize = detector.settings['subsample_size']
    rotspread = detector.settings.get('rotation_spreading_radius', 0)
    cb = detector.settings.get('crop_border')
    setts = dict(spread_radii=radii, subsample_size=psize, rotation_spreading_radius=rotspread, crop_border=cb)

    for neg in itr.islice(gen, PER_IMAGE):
        #images.append(neg)
        feat = detector.descriptor.extract_features(neg, settings=setts)
        neg_feats.append(feat)

    return neg_feats

def get_strong_fps(detector, i, fileobj, threshold, mixcomp):
    topsy = [[] for k in xrange(detector.num_mixtures)]
    #for i, fileobj in enumerate(itr.islice(gen, COUNT)):
    ag.info('{0} Farming {1}'.format(i, fileobj.img_id))
    #img = gv.img.load_image(img_fn)
    img = gv.img.load_image(fileobj.path)
    grayscale_img = gv.img.asgray(img)

    for m in xrange(detector.num_mixtures):
        bbobjs = detector.detect_coarse(grayscale_img, fileobj=fileobj, mixcomps=[m], use_padding=False, use_scale_prior=False, cascade=True, discard_weak=True, more_detections=True)
        # Add in img_id
        for bbobj in bbobjs:
            bbobj.img_id = fileobj.img_id

            #array.append(bbobj.X)

            if bbobj.confidence > threshold: 
                #if len(topsy[m]) < TOP_N:
                #    heapq.heappush(topsy[m], bbobj)
                #else:
                #    heapq.heappushpop(topsy[m], bbobj)
                topsy[m].append(bbobj)

    return topsy

def get_strong_fps_single(detector, i, fileobj, threshold, mixcomp):
    topsy = []
    ag.info('{0} Farming {1}'.format(i, fileobj.img_id))
    img = gv.img.load_image(fileobj.path)
    grayscale_img = gv.img.asgray(img)

    bbobjs = detector.detect_coarse(grayscale_img, fileobj=fileobj, mixcomps=[mixcomp], use_padding=False, use_scale_prior=False, cascade=True, discard_weak=True, more_detections=True, farming=True)
    for bbobj in bbobjs:
        bbobj.img_id = fileobj.img_id
        if bbobj.confidence > threshold: 
            topsy.append(bbobj)

    return topsy


if gv.parallel.main(__name__):
    from settings import argparse_settings
    settings = argparse_settings("Train real-valued detector")
    dsettings = settings['detector']

    #descriptor = gv.load_descriptor(gv.RealDetector.DESCRIPTOR, sett)
    descriptor = gv.load_descriptor(settings)
    detector = gv.RealDetector(descriptor, dsettings)

    all_files = sorted(glob.glob(os.path.expandvars(dsettings['train_dir'])))
    assert len(all_files) > 0, 'No files'
    random.seed(0)
    random.shuffle(all_files)
    files = all_files[:dsettings.get('train_limit')]
    neg_files = sorted(glob.glob(os.path.expandvars(dsettings['neg_dir'])))[:dsettings.get('neg_limit')]
    #pos_images = []
    image_size = detector.settings['image_size']

    print('Edge type:', descriptor.settings.get('edge_type', '(none)'))

    # Extract clusters (manual or through EM)
    ##############################################################################
    detector, comps = cluster(detector, files)

    bbs = calc_bbs(detector)

    print(bbs)

    print(comps)


    # Avoids sending the detector once for each image. This could be
    # optimized away with a more clever slution in gv.parallel. Think:
    #
    # gv.parallel.smartmap_unordered(get_positives, detector, batches)
    #
    # It detects detector is not iterable (or wrapped in something),
    # and sends it only once to each worker.
    print('Starting')

    def make_empty_lists(N):
        return map(list, [[]] * N) 

    all_pos_feats = make_empty_lists(detector.num_mixtures)
    all_alphas = make_empty_lists(detector.num_mixtures)
    alphas = make_empty_lists(detector.num_mixtures)

    if detector.settings.get('superimpose'):
        print('Fetching positives (background superimposed)')
        # The positives don't have background, superimpose onto random background
        argses = [(m, settings, bbs[m], list(np.where(comps == m)[0]), files, neg_files, settings['detector'].get('stand_multiples', 1)) for m in range(detector.num_mixtures)]        
        for m, _, pos_feats_chunk, alpha_maps, extra in itr.starmap(get_pos_and_neg, argses):
            all_alphas[m] += alpha_maps
            all_pos_feats[m] += list(pos_feats_chunk)
            print('Chunk size', len(pos_feats_chunk), 'mixture', m)

            #detector.extra['concentrations'].append(extra.get('concentrations', {}))

        for m in xrange(detector.num_mixtures):
            alpha = np.mean(all_alphas[m], axis=0)
            alphas[m] = alpha

        print('SUPPORT')
        detector.support = alphas

        # Update image size
    else:
        print('Fetching positives (background included)')
        #BATCH_SIZE = 5
        # The positives already have background 
        #batches = np.array_split(files, len(files)//BATCH_SIZE)
        #args = itr.izip(itr.repeat(detector), batches)

        argses = [(m, settings, list(np.where(comps == m)[0]), files) for m in range(detector.num_mixtures)]        
        for m, new_pos_feats in gv.parallel.starmap_unordered(get_positives, argses):
            all_pos_feats[m] += new_pos_feats 

    print('Finished')
    feats = []

    #images = pos_images[:]

    detector.svms = make_empty_lists(detector.num_mixtures)
    detector.kernel_sizes = make_empty_lists(detector.num_mixtures)

    for m in xrange(detector.num_mixtures):
        pos_feats = all_pos_feats[m] 
        pos_labels = [1] * len(pos_feats) 

        neg_feats = []
        neg_labels = []

        source = detector.settings.get('negative_source', 'neg-dir')
        if source.startswith('voc-train-non-'):
            obj_class = source.split('-')[-1] 
            print('Taking negatives from voc train, without class', obj_class)
            gen = gv.voc.gen_negative_files(obj_class, 'train')
            #print('negatives', len([im for im in gen]))
        else:
            print('Taking negatives from neg_dir')
            gen = itr.cycle(gv.datasets.ImgFile(path=fn, img_id=os.path.basename(fn)) for fn in neg_files)
            
        gen = itr.cycle(gen)
        index = 0

        if 1:
            #neg_files_segment = itr.islice(gen, dsettings['neg_limit'])
            count = 0

            argses = [(detector, i, fileobj, gv.bb.size(bbs[m])) for i, fileobj in enumerate(itr.islice(gen, dsettings['neg_limit']//PER_IMAGE))]
            for new_neg_feats in gv.parallel.starmap_unordered(get_fps, argses):
                neg_feats += new_neg_feats 
                neg_labels += [0] * len(new_neg_feats)

            #labels = pos_labels + [0] * count

        else:
            # Add negatives
            assert len(neg_files) > 0, 'No negative files'
            files += neg_files
            labels += [0] * len(neg_files)


        feats = pos_feats + neg_feats
        labels = pos_labels + neg_labels


        feats = np.asarray(feats)
        labels = np.asarray(labels)

        with gv.Timer('Saving training data'):
            np.savez('/var/tmp/d/training_data.npz', feats=feats, labels=labels)

        print("Training with {total} (pos = {pos})".format(total=len(feats), pos=np.sum(np.asarray(labels)==1)))
        print("feats", feats.shape)
        with gv.Timer('Training SVM'):
            svms, kernel_sizes = detector.train_from_features(feats, labels, save=False)

        detector.svms[m] = svms[0]
        detector.kernel_sizes[m] = kernel_sizes[0]

        farming_image_counts = detector.settings.get('farming_image_counts', [1000])
        farming_top_counts = detector.settings.get('farming_top_counts', [4000])
        assert len(farming_image_counts) == len(farming_top_counts)

        #neg_files_loop = itr.cycle(neg_files)
        print("Farming loop...")

        feats = list(feats)
        labels = list(labels)

        detectors = []
        detectors.append(detector)
        cascades = []
        detector.extra['cascades'] = []
        cur_detector = detector
        for loop, (N, TOP) in enumerate(zip(farming_image_counts, farming_top_counts)): 
            feats = pos_feats + neg_feats
            labels = pos_labels + neg_labels
            #th = -0.75
            th = -np.inf 

            neg_files_segment = itr.islice(gen, N)

            argses = [(cur_detector, index+i, fileobj, th, m) for i, fileobj in enumerate(neg_files_segment)] 
            index += N 

            topsy = list(gv.parallel.starmap_unordered(get_strong_fps_single, argses))
            confs = np.asarray([bbobj.confidence for topsy_m in topsy for bbobj in topsy_m])
            from scipy.stats.mstats import scoreatpercentile

            confs = np.sort(confs)

            # Maybe a different th0 for each component?
            #th0 = float(scoreatpercentile(confs, 75))

            if len(confs) >= TOP: 
                th0 = confs[-TOP]
            else:
                th0 = confs[0]

            negs = []
            print("Starting...")
            #for topsy in gv.parallel.imap_unordered(get_strong_fps_star, argses):
            for topsy_m in topsy:
                for bbobj in topsy_m:
                    if bbobj.confidence >= th0:
                        feats.append(bbobj.X)
                        labels.append(0)

            #detector0 = deepcopy(cur_detector)
            detector0 = cur_detector
            print("Training with {total} ({pos})".format(total=len(feats), pos=np.sum(np.asarray(labels)==1)))
            with gv.Timer('Training SVM'):
                svms, kernel_sizes = detector0.train_from_features(feats, labels, save=False)

            count = np.sum(np.asarray(labels)==0)

            # Do not cascade, instead override the detector
            detector.svms[m] = svms[0]
            detector.kernel_sizes[m] = kernel_sizes[0]

            with gv.Timer('Saving training data'):
                np.savez('/var/tmp/d/training_data-iter{}.npz'.format(loop), feats=feats, labels=labels)

            if 0:
                detectors.append(detector0) 
                info = dict(th=th0, svms=svms, count=count)
                #cascades.append(info)
                detector.extra['cascades'].append(info)

                cur_detector = detector0

        if 0:
            detector.svms = detector.extra['cascades'][0]['svms']
            detector.extra['cascades'] = []
        else:
            detector = detectors[0]
        #detector.extra['cascades'] = cascades

            #print "Training on {0} files".format(len(files))
            #print "{0} pos / {1} neg".format(np.sum(labels == 1), np.sum(labels == 0))
            # Re-train the detector now
            #labels +=
            #detector.train_from_image_data(images, labels)


        #labels = np.asarray(labels)
            
        #files = files[:10]

    detector.save(dsettings['file'])
    print('Counts', [x['count'] for x in detector.extra['cascades']])
    print('Th', [x['th'] for x in detector.extra['cascades']])
    print("Saved, exiting")

