from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='Test response of model')
parser.add_argument('results', metavar='<file>', nargs=1, type=argparse.FileType('rb'), help='Filename of results file')
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')
parser.add_argument('new_model', metavar='<output model file>', type=argparse.FileType('wb'), help='Filename of model file')
parser.add_argument('-o', '--output', type=str, default=None)

args = parser.parse_args()
model_file = args.model
new_model_file = args.new_model
results_file = args.results[0]

import numpy as np
import gv
import itertools
import copy

detector = gv.Detector.load(model_file)
data = np.load(results_file)

#p, r = data['precisions'], data['recalls']
detections = data['detections']
mixcomps = detections['mixcomp'].max() + 1
bkgcomps = detections['bkgcomp'].max() + 1

tps_fps = np.zeros((mixcomps, bkgcomps, 2))

#mixcomp = 3
#mixcomp = args.mixcomp

I = detector.extra['bkg_mixtures']
new_I = copy.deepcopy(I)

#gs = gridspec.GridSpec(mixcomps, 2, width_ratios=[7,1])
for mixcomp, bkgcomp in itertools.product(xrange(mixcomps), xrange(bkgcomps)):
    mydets = detections[(detections['mixcomp'] == mixcomp) & (detections['bkgcomp'] == bkgcomp)]
    I0 = I[mixcomp][bkgcomp]

    tps_fps[mixcomp,bkgcomp,0] = mydets[mydets['correct'] == 0].size
    tps_fps[mixcomp,bkgcomp,1] = mydets[mydets['correct'] == 1].size

    llhs = (mydets[mydets['correct'] == 0]['confidence'] - 100) * I0['std'] + I0['mean']
    mean = np.mean(llhs)
    std = np.std(llhs)
    samples = llhs.size

    new_I[mixcomp][bkgcomp]['mean'] = mean
    new_I[mixcomp][bkgcomp]['std'] = std

    print '({mixcomp}, {bkgcomp}) mean={mean}, std={std} (samples={samples})'.format(
            mixcomp=mixcomp, bkgcomp=bkgcomp, mean=mean, std=std, samples=samples)


detector.extra['bkg_mixtures'] = new_I
detector.save(new_model_file)
