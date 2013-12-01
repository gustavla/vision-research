
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_model', metavar='<input model file>', type=argparse.FileType('rb'), help='Filename of input model file')
parser.add_argument('output_model', metavar='<output model file>', type=argparse.FileType('wb'), help='Filename of model model file')

args = parser.parse_args()

import numpy as np
import gv

from train_superimposed import get_key_points

detector = gv.Detector.load(args.input_model)

# Re-train the SVM, using key points

from sklearn.svm import SVC

K = detector.num_mixtures
#detector.indices2 = None
detector.indices2 = []
for k in xrange(K):
    X = detector.extra['data_x'][k]  
    y = detector.extra['data_y'][k]

    clf = detector.clfs[k]['svm']

    sh = detector.kernel_templates[k][0].shape

    clf3 = SVC(C=0.00005, kernel='linear')
    clf3.fit(X, y)

    # Select key points from SVM coefficients
    if 0:
        II = np.argsort(clf3.coef_.ravel())
        ii = np.asarray(map(lambda x: np.unravel_index(x, sh), II[-1000:]))

        #coef3d = clf3.coef_[0].reshape(sh)
        #ii = get_key_points(coef3d, suppress_radius=1)
        detector.indices2.append(ii)

        X2 = np.zeros((X.shape[0], ii.shape[0]))
        #import pdb; pdb.set_trace()
        for i, Xi in enumerate(X):
            X2[i] = Xi.ravel()[np.ravel_multi_index(ii.T, sh)]

        clf2 = SVC(C=0.0005, kernel='linear')
        clf2.fit(X2, y)

    #import pdb; pdb.set_trace()
    #clf2.predict(X2) == 

    #clf3.coef_ = clf3.coef_

    # Replace 
    detector.clfs[k]['svm'] = clf3 

detector.settings['max_size'] = 700
#detector.settings['scale_factor'] = 2**(1/5.)

detector.save(args.output_model)
