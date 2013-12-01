

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

ag.set_verbose(True)

#descriptor = gv.load_descriptor(gv.RealDetector.DESCRIPTOR, sett)
descriptor = gv.load_descriptor(sett)
detector = gv.RealDetector(descriptor, dsettings)

all_files = sorted(glob.glob(os.path.expandvars(dsettings['train_dir'])))
random.seed(0)
random.shuffle(all_files)
files = all_files[:dsettings.get('train_limit')]
labels = [1] * len(files)
images = []

for fn in files:
    im = gv.img.asgray(gv.img.load_image(fn))
    images.append(im)

from train_superimposed import generate_random_patches

feats = []

if 1:
    neg_files = sorted(glob.glob(os.path.expandvars(dsettings['neg_dir'])))
    gen = generate_random_patches(neg_files, dsettings['image_size'], per_image=5)
    count = 0
    for neg in gen:
        print count
        images.append(neg)
        count += 1
        if count == dsettings['neg_limit']:
            break

    labels += [0] * count

else:
    # Add negatives
    neg_files = sorted(glob.glob(os.path.expandvars(dsettings['neg_dir'])))[:dsettings.get('neg_limit')]
    files += neg_files
    labels += [0] * len(neg_files)

labels = np.asarray(labels)
    
print "Training on {0} files".format(len(files))
print "{0} pos / {1} neg".format(np.sum(labels == 1), np.sum(labels == 0))
#files = files[:10]

detector.train_from_image_data(images, labels)

detector.save(dsettings['file'])

