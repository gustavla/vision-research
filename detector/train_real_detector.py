

from settings import argparse_settings
sett = argparse_settings("Train real-valued detector")
dsettings = sett['detector']

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
import numpy as np
import amitgroup as ag
import random
import itertools as itr
from copy import deepcopy
from train_superimposed import generate_random_patches

ag.set_verbose(True)

PER_IMAGE = 5

def get_fps(detector, i, fileobj):
    ag.info('{0} Initial processing {1}'.format(i, fileobj.img_id))
    gen = generate_random_patches([fileobj.path], dsettings['image_size'], per_image=PER_IMAGE)
    neg_feats = []

    for neg in itr.islice(gen, PER_IMAGE):
        #images.append(neg)
        feat = detector.descriptor.extract_features(neg)
        neg_feats.append(feat)

    return neg_feats

def get_fps_star(args):
    return get_fps(*args)

def get_strong_fps(detector, i, fileobj, threshold):
    topsy = [[] for k in xrange(detector.num_mixtures)]
    #for i, fileobj in enumerate(itr.islice(gen, COUNT)):
    ag.info('{0} Farming {1}'.format(i, fileobj.img_id))
    #img = gv.img.load_image(img_fn)
    img = gv.img.load_image(fileobj.path)
    grayscale_img = gv.img.asgray(img)

    for m in xrange(detector.num_mixtures):
        bbobjs = detector.detect_coarse(grayscale_img, fileobj=fileobj, mixcomps=[m], use_padding=False, use_scale_prior=False, cascade=False, more_detections=True)
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

def get_strong_fps_star(args):
    return get_strong_fps(*args)


if gv.parallel.main(__name__):
    #descriptor = gv.load_descriptor(gv.RealDetector.DESCRIPTOR, sett)
    descriptor = gv.load_descriptor(sett)
    detector = gv.RealDetector(descriptor, dsettings)

    all_files = sorted(glob.glob(os.path.expandvars(dsettings['train_dir'])))
    random.seed(0)
    random.shuffle(all_files)
    files = all_files[:dsettings.get('train_limit')]
    pos_labels = [1] * len(files)
    pos_images = []
    pos_feats = []

    for fn in files:
        im = gv.img.asgray(gv.img.load_image(fn))
        pos_images.append(im)

        feat = detector.descriptor.extract_features(im)
        pos_feats.append(feat)

    feats = []

    images = pos_images[:]

    neg_feats = []
    neg_labels = []

    contest = 'voc'
    obj_class = 'car'
    gen = gv.voc.gen_negative_files(obj_class, 'train')
    gen = itr.cycle(gen)
    index = 0

    if 1:
        #neg_files_segment = itr.islice(gen, dsettings['neg_limit'])
        count = 0

        argses = [(detector, i, fileobj) for i, fileobj in enumerate(itr.islice(gen, dsettings['neg_limit']//PER_IMAGE))]
        for new_neg_feats in gv.parallel.imap_unordered(get_fps_star, argses):
            neg_feats += new_neg_feats 
            neg_labels += [0] * len(new_neg_feats)

        #labels = pos_labels + [0] * count

    else:
        # Add negatives
        neg_files = sorted(glob.glob(os.path.expandvars(dsettings['neg_dir'])))[:dsettings.get('neg_limit')]
        files += neg_files
        labels += [0] * len(neg_files)


    feats = pos_feats + neg_feats
    labels = pos_labels + neg_labels

    print "Training with {total} ({pos})".format(total=len(feats), pos=np.sum(np.asarray(labels)==1))
    detector.train_from_features(feats, labels)

    N_FARMING_ITER = 1

    #neg_files_loop = itr.cycle(neg_files)
    print "Farming loop..."

    detectors = []
    detectors.append(detector)
    cascades = []
    cur_detector = detector
    for loop in xrange(N_FARMING_ITER):
        feats = pos_feats #+ neg_feats
        labels = pos_labels #+ neg_labels
        #th = -0.75
        th = 0.0

        # Process files
        #for i in xrange(100):
            #topsy = get_strong_fps 
        #argses = itr.islice(files, 
        N = 100
        neg_files_segment = itr.islice(gen, N)

        argses = [(cur_detector, index+i, fileobj, th) for i, fileobj in enumerate(neg_files_segment)] 
        index += N 

        topsies = list(gv.parallel.imap_unordered(get_strong_fps_star, argses))
        confs = np.asarray([bbobj.confidence for topsy in topsies for topsy_m in topsy for bbobj in topsy_m])
        from scipy.stats.mstats import scoreatpercentile

        # Maybe a different th0 for each component?
        th0 = float(scoreatpercentile(confs, 75))

        negs = []
        print "Starting..."
        #for topsy in gv.parallel.imap_unordered(get_strong_fps_star, argses):
        for topsy in topsies:
            for m in xrange(cur_detector.num_mixtures):
                for bbobj in topsy[m]:
                    if bbobj.confidence >= th0:
                        feats.append(bbobj.X)
                        labels.append(0)

        detector0 = deepcopy(cur_detector)
        print "Training with {total} ({pos})".format(total=len(feats), pos=np.sum(np.asarray(labels)==1))
        svms, kernel_sizes = detector0.train_from_features(feats, labels)

        count = np.sum(np.asarray(labels)==0)

        detectors.append(detector0) 
        cascades.append(dict(th=th0, svms=svms, count=count))

        cur_detector = detector0

    detector = detectors[0]
    detector.extra['cascades'] = cascades

        #print "Training on {0} files".format(len(files))
        #print "{0} pos / {1} neg".format(np.sum(labels == 1), np.sum(labels == 0))
        # Re-train the detector now
        #labels +=
        #detector.train_from_image_data(images, labels)


    #labels = np.asarray(labels)
        
    #files = files[:10]

    detector.save(dsettings['file'])
    print "Saved, exiting"

