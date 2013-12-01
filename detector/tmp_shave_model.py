
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_model', metavar='<input model file>', type=argparse.FileType('rb'), help='Filename of input model file')
parser.add_argument('output_model', metavar='<output model file>', type=argparse.FileType('wb'), help='Filename of model model file')

args = parser.parse_args()

import numpy as np
import gv

detector = gv.Detector.load(args.input_model)

del detector.extra['poss']
del detector.extra['negs']
origs = detector.descriptor.extra.get('originals')
if origs is not None:
    origs = [o[:20] for o in origs]
    detector.descriptor.extra['originals'] = origs

for m in xrange(detector.num_mixtures):
    f = detector.extra['svms'][m]
    f['intercept'] = f['svm'].intercept_
    f['weights'] = f['svm'].coef_
    del f['svm']

detector.save(args.output_model)
#del detector.extra['negs']

