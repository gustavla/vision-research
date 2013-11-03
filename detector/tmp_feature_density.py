from __future__ import division
import os
import gv
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Test response of model')
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')

args = parser.parse_args()
model_file = args.model

detector = gv.Detector.load(model_file)

psize = detector.settings['subsample_size']
radii = detector.settings['spread_radii']

path = os.path.expandvars('$VOC_DIR/JPEGImages/001119.jpg')

import skimage.io

im = skimage.io.imread(path)
print im.sum()
np.save('im.npy', im)

im = gv.img.asgray(gv.img.load_image(path))
print im.sum()
spread_feats = detector.descriptor.extract_features(im, dict(spread_radii=radii, preserve_size=True))
feats = gv.sub.subsample(spread_feats, psize)

print feats.sum()
