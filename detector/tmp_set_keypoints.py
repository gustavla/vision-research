
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input_model', metavar='<input model file>', type=argparse.FileType('rb'), help='Filename of input model file')
parser.add_argument('output_model', metavar='<output model file>', type=argparse.FileType('wb'), help='Filename of model model file')

args = parser.parse_args()

import numpy as np
import gv
import gv.fast
import itertools as itr

from train_superimposed import get_key_points

detector = gv.Detector.load(args.input_model)

# Re-train the SVM, using key points

from sklearn.svm import SVC

np.seterr(all='ignore')

M = detector.num_mixtures

def array_argmax(array):
    return np.unravel_index(array.argmax(), array.shape)

def myprod(shape):
    return itr.product(*[xrange(i) for i in shape])

indices = []

threshold = 0.1

detector.indices = []
detector.standardization_info = []

from sklearn import linear_model
from skimage import morphology, feature

EACH = 40 

for m in xrange(M):
    P = detector.extra['poss'][m]
    #N = np.asarray([bbobj.X for bbobj in detector.extra['negs'][m]])

    #X_shaped = np.conatenate([P, N])
    #X = X_shaped.reshape((X_shaped[0], -1)).astype(np.float64)
    #y = np.concatenate([np.ones(len(P)), np.zeros(len(N))])
    # Get the keypoints  

    w = detector.weights(m)
    # Remove twin-peaks by small noise
    w += np.random.normal(scale=1e-5, size=w.shape)
    

    part_indices = []
    for f in xrange(detector.num_features):
        II = feature.peak_local_max(np.fabs(w[...,f]), threshold_rel=0, min_distance=1, footprint=morphology.disk(3), labels=(1+(w[...,f]>0)), exclude_border=False)
        
        part_indices.append(np.hstack([II, f*np.ones((II.shape[0], 1))]))

    indices = np.concatenate(part_indices)

    if 0:
        w_p = w.copy()
        w_p[w < 0] = 0.0
        w_n = w.copy()
        w_n[w > 0] = 0.0

        indices_p = get_key_points(w_p, 4)
        indices_n = get_key_points(w_n, 4)
        
        indices = np.concatenate([indices_p, indices_n])

    #alphas, _, coefs = linear_model.lars_path(X, y, method='lar', verbose=True)

        
    
    if 0:
        loop = 0
        while True: 
            I = array_argmax(w)
            if w[I] == 0.0:
                break

            A = P[:,I[0],I[1],I[2]]
            M = np.zeros_like(w)
            
            # Now check the correlation with this point and the rest
            #for i, j, k in myprod(w.shape):
                #M[i,j,k] = np.corrcoef(A, P[:,i,j,k])[1,0]
            #M[np.isnan(M)] = 0.0
            M = gv.fast.correlate_abunch(A, P)
            M[I] = 1.0

            II = [np.unravel_index(index, M.shape) for index in M.ravel().argsort()[-EACH:]]

            for index in II:
                w[tuple(index)] = 0.0

            # Start suppressing
            # float(np.sum((np.fabs(M) > threshold) & (w != 0))
            print '{0:4} Suppressing {1:3.2f}'.format(len(indices), EACH),
            print ' w: {0:.02f} max {1:3.2f} {2}'.format(float(np.mean(w > 0) * 100), w[I], I)
            #w[np.fabs(M) > threshold] = 0.0

            indices.append(I)
            #print I
            loop += 1
            #if loop >= 40:
                #import pdb; pdb.set_trace()
                #break
    #print 'indices', len(indices)

    eps = detector.settings['min_probability']
    clipped_bkg = np.clip(detector.fixed_spread_bkg[m][0], eps, 1 - eps)

    # Re-standardize
    llh_mean = 0.0
    llh_var = 0.0
    for index in indices:
        part = index[-1]
        mvalue = clipped_bkg[...,part].mean()
        llh_mean += mvalue * w[tuple(index)]
        llh_var += mvalue * (1 - mvalue) * w[tuple(index)]**2

    info = {}
    info['mean'] = llh_mean 
    info['std'] = np.sqrt(llh_var)

    detector.indices.append([indices])
    detector.standardization_info.append([info])

print 'shapes', map(np.shape, detector.indices)
    
detector.save(args.output_model)
