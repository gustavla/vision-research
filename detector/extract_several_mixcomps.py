from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='Create a single component model')
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')
parser.add_argument('output', metavar='<new model file>', type=argparse.FileType('wb'), help='Filename of new model file')
parser.add_argument('mixcomps', nargs='+', type=int, help='Select mixture components')

args = parser.parse_args()
model_file = args.model
output_file = args.output
mixcomps = args.mixcomps

import numpy as np
import gv

d = gv.Detector.load(model_file)

objs = [
    d.kernel_templates, 
    d.fixed_spread_bkg,
    d.fixed_spread_bkg2,
    d.kernel_sizes, 
    d.support, 
    d.standardization_info, 
    d.standardization_info2,
    d.indices,
    d.clfs,
]

for obj in objs:
    if obj is not None:
        obj[:] = [obj[i] for i in mixcomps]

#d.kernel_templates = d.kernel_templates[mixcomp:mixcomp+1]
#d.kernel_sizes = d.kernel_sizes[mixcomp:mixcomp+1]
#d.support = d.support[mixcomp:mixcomp+1]
#d.standardization_info = d.standardization_info[mixcomp:mixcomp+1]
#d.clfs = d.clfs[mixcomp:mixcomp+1]
d.num_mixtures = len(mixcomps) 

d.save(output_file)
