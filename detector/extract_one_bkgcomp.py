from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='Create a single component model')
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')
parser.add_argument('output', metavar='<new model file>', type=argparse.FileType('wb'), help='Filename of new model file')
parser.add_argument('bkgcomp', metavar='BKGCOMP', type=int, help='Select background component')

args = parser.parse_args()
model_file = args.model
output_file = args.output
bkgcomp = args.bkgcomp

import numpy as np
import gv

d = gv.Detector.load(model_file)

for mixcomp in xrange(d.num_mixtures):
    d.kernel_templates[mixcomp] = d.kernel_templates[mixcomp][bkgcomp:bkgcomp+1]
    #d.kernel_sizes[mixcomp] = d.kernel_sizes[mixcomp][bkgcomp:bkgcomp+1]
    d.fixed_spread_bkg[mixcomp] = d.fixed_spread_bkg[mixcomp][bkgcomp:bkgcomp+1]
    d.standardization_info[mixcomp] = d.standardization_info[mixcomp][bkgcomp:bkgcomp+1]

d.settings['num_bkg_mixtures'] = 1

d.save(output_file)
