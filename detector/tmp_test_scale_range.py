from __future__ import division
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')

args = parser.parse_args()

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pylab as plt
import gv
import amitgroup as ag
import numpy as np

detector = gv.Detector.load(args.model)
detector.TEMP_second = True

mixcomp = 2 

# Image to do detection in
cad_im0 = gv.img.load_image('/var/tmp/matlab/xi3zao3-car-centered7/5-view000_car02.png')
#bkg_im = 0.5 * np.ones(I
factors = np.linspace(0.5, 2.0, 40)
scores = np.zeros(len(factors))

for i, factor in enumerate(factors):
    #cad_im = gv.img.resize_with_factor_new(cad_im0, factor)
    #cad_feat = detector.extract_spread_features(cad_im)

    bbs, resmap, bkgcomp, spread_feats, img_resized = detector.detect_coarse_single_factor(cad_im0, factor, mixcomp)

    #scores[i] = resmap.max()
    scores[i] = bbs[0].score

plt.plot(factors, scores)
plt.show()
