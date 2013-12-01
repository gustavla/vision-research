import argparse
from settings import load_settings

parser = argparse.ArgumentParser(description='Train mixture model on edge data')
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')
parser.add_argument('output_model', metavar='<output model file>', type=argparse.FileType('wb'), help='Filename of output model file')
parser.add_argument('--limit', type=int, default=10)

args = parser.parse_args()

import gv
import os
import os.path
import glob
import numpy as np
import amitgroup as ag
from scipy.stats import mstats

detector = gv.Detector.load(args.model)

# Turn off standardization
detector.settings['testing_type'] = None 

path = os.path.expandvars(detector.settings['neg_dir'])
files = sorted(glob.glob(path))

files = files[:args.limit]

all_resmaps = []

for f in files:
    print 'Processing', f

    im = gv.img.load_image(f)
    im = gv.img.asgray(im)

    bbs, resmaps = detector.detect_coarse(im, return_resmaps=True)

    all_resmaps.append(resmaps)

    #im = gv.img.resize_with_factor_new(im, factor)

factors = all_resmaps[0].keys()

agg_data = {}
new_info = []

for factor in factors:
    agg_data[factor] = []
    for mixcomp in xrange(detector.num_mixtures):
        values = np.concatenate([resmap[factor][mixcomp].ravel() for resmap in all_resmaps])
        agg_data[factor].append(values)


#np.save('agg_data.npy', agg_data)
#import pdb; pdb.set_trace()

new_info = []

for mixcomp in xrange(detector.num_mixtures):
    #values = values_array[mixcomp]
    #info.append(d)
    factor_info = []
        
    for factor, values_array in sorted(agg_data.items()):
        values = values_array[mixcomp] 
        d = dict(neg_llhs=values, neg_mean=values.mean(), neg_std=values.std())
        factor_info.append(d) 

    detector.standardization_info[mixcomp][0]['factor_info'] = factor_info

    #new_info.append(info)

#detector.standardization_info = new_info
detector.save(args.output_model)
