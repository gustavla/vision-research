
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_model', metavar='<input model file>', type=argparse.FileType('rb'), help='Filename of input model file')
parser.add_argument('output_model', metavar='<output model file>', type=argparse.FileType('wb'), help='Filename of model model file')

args = parser.parse_args()

import numpy as np
import gv
from sklearn.svm import SVC

detector = gv.Detector.load(args.input_model)

C = 5e-4

for m in xrange(detector.num_mixtures):
    all_pos_X0 = [X.ravel() for X in detector.extra['poss'][m]]
    all_neg_X0 = [bbobj.X.ravel() for bbobj in detector.extra['negs'][m]]

    X = np.concatenate([all_pos_X0, all_neg_X0])  
    y = np.concatenate([np.ones(len(all_pos_X0)), np.zeros(len(all_neg_X0))])

    print X.shape
    print y.shape

    svm = SVC(C, kernel='linear') 
    svm.fit(X, y)

    detector.extra['svms'][m]['svm'] = svm

del detector.extra['poss']
del detector.extra['negs']

detector.save(args.output_model)
