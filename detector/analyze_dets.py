from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='Test response of model')
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')
parser.add_argument('results', metavar='<file>', type=argparse.FileType('rb'), help='Filename of results file')
parser.add_argument('--limit', type=int, default=None)
#parser.add_argument('output_dir', metavar='<output folder>', nargs=1, type=str, help="Output path")
#parser.add_argument('img_id', metavar='<image id>', type=int, help='ID of image in VOC repository')

args = parser.parse_args()
model_file = args.model
results_file = args.results
limit = args.limit
#output_dir = args.output_dir[0]
#img_id = args.img_id

import matplotlib.pylab as plt
import gv
import numpy as np
import amitgroup as ag
from plotting import plot_image
import os

detector = gv.Detector.load(model_file)

data = np.load(results_file)

#p, r = data['precisions'], data['recalls']
detections = data['detections']
detections.sort(order='confidence')
detections = detections[::-1]

# TODO:
try:
    contest = str(data['contest'])
    obj_class = data['obj_class']
except KeyError:
    contest = 'voc'
    obj_class = 'car'

def _classify(neg_feats, pos_feats, mixture_params):
    from scipy.stats import beta
    M = len(mixture_params)
    collapsed_feats = np.apply_over_axes(np.mean, neg_feats, [0, 1]).ravel()
    collapsed_feats = np.clip(collapsed_feats, 0.01, 1-0.01)
    D = collapsed_feats.shape[0]
    
    qlogs = np.zeros(M)
    for m in xrange(M):
        #v = qlogs[m] 
        v = 0.0
        for d in xrange(D):
            v += beta.logpdf(collapsed_feats[d], mixture_params[m,d,0], mixture_params[m,d,1])
        qlogs[m] = v

    #bkg_id = qlogs.argmax()
    #return bkg_id
    return qlogs

psize = detector.settings['subsample_size']
radii = detector.settings['spread_radii']

kernels = detector.prepare_kernels(None, settings=dict(spread_radii=radii, subsample_size=psize))
all_bkg = detector.bkg_model(None, spread=True)
eps = detector.settings['min_probability']
#bkg = np.clip(bkg, eps, 1 - eps)

analyze_mixcomp = -1 
Y = np.zeros((2,) + kernels[analyze_mixcomp][0].shape, dtype=np.uint32)
totals = np.zeros(2, dtype=np.uint32)
all_dists = [[], []] 

all_part_probs = [[], []]

for i, det in enumerate(detections[:limit]):
    bb = (det['top'], det['left'], det['bottom'], det['right'])
    k = det['mixcomp']
    m = det['bkgcomp']
    bbobj = gv.bb.DetectionBB(bb, score=det['confidence'], confidence=det['confidence'], mixcomp=k, correct=det['correct'], index_pos=(det['index_pos0'], det['index_pos1']), scale=det['scale'], img_id=det['img_id'])

    #im = gv.img.load_image(fileobj.path) 
    #im = gv.img.asgray(im)
    #im = gv.img.resize_with_factor_new(im, 1/det['scale'])

    kern = kernels[k][m]
    bkg = all_bkg[k][m]
    kern = np.clip(kern, eps, 1 - eps)
    bkg = np.clip(bkg, eps, 1 - eps)

    X = gv.datasets.extract_features_from_bbobj(bbobj, detector, contest, obj_class, kern.shape[:2])

    weights = np.log(kern / (1 - kern) * ((1 - bkg) / bkg))

    #print 'X', X.shape
    if k == 2:
        part_probs = np.apply_over_axes(np.mean, X, [0, 1])[0,0]
        all_part_probs[det['correct']].append(part_probs)

    # bkg levels
    if 0:
        #Xdist = np.apply_over_axes(np.mean, X, [0, 1]).ravel()

        params = detector.bkg_mixture_params

        from gv.fast import bkg_model_dists
        
        dists = -bkg_model_dists(X, detector.bkg_mixture_params, X.shape[:2], padding=0, inner_padding=-2)[0,0]
        #qlogs 
        all_dists[det['correct']].append(dists.max())
    else:
        dists = None

    #if analyze_mixcomp == k or analyze_mixcomp == -1:
        #Y[det['correct']] += X
        #totals[det['correct']] += 1

    from gv.fast import multifeature_correlate2d_with_indices, multifeature_correlate2d
    #R = float((X * weights).sum())
    #Rst = (R - detector.fixed_train_mean[k]) / detector.fixed_train_std[k]

    if detector.indices is not None:
        R = multifeature_correlate2d_with_indices(X, weights.astype(np.float64), detector.indices[k][m])[0,0]
    else:
        R = multifeature_correlate2d(X, weights.astype(np.float64))[0,0]
    
    from gv.fast import nonparametric_rescore
    Rsts = np.zeros(detector.num_bkg_mixtures)
    for m in xrange(detector.num_bkg_mixtures):
        Rarray = R * np.ones((1, 1))
        info = detector.standardization_info[k][m]
        nonparametric_rescore(Rarray, info['start'], info['step'], info['points'])
        #Rst = R
        Rsts[m] = Rarray[0,0]

    Rst = Rsts[det['bkgcomp']]
    
    # Replace bounding boxes with this single one
    #fileobj.boxes[:] = [bbobj]
    
    fn = 'det-{0}.png {1}'.format(i, bbobj.img_id)
    #print '{batsu} {fn}: {standardized:.2f} {raw:.2f} ({mixcomp}, {bkgcomp})'.format(fn=fn, standardized=Rst, raw=det['confidence'], batsu=['X', '.'][det['correct']], mixcomp=det['mixcomp'], bkgcomp=det['bkgcomp'])
    print '{batsu} {fn}: {standardized:.2f} {raw:.2f} ({mixcomp}, {bkgcomp}) dist={dist}'.format(
        fn=fn, standardized=Rst, raw=det['confidence'], batsu=['X', '.'][det['correct']], mixcomp=det['mixcomp'], bkgcomp=det['bkgcomp'], dist=dists,
    )
    #import IPython
    #IPython.embed()
    #print 'scale', np.log(det['scale'])/np.log(2)
    #print i0, j0

#Z = Y.astype(np.float64) / totals.reshape((-1,) + (1,)*(Y.ndim-1))
#G = (Z * weights)

all_part_probs[0] = np.asarray(all_part_probs[0])
all_part_probs[1] = np.asarray(all_part_probs[1])

if 0: 
    Gmeans = [G[i].mean(axis=-1) for i in xrange(2)]
    m = max(np.fabs(Gmeans[0]).max(), np.fabs(Gmeans[1]).max())

    plt.subplot(211)
    plt.imshow(Gmeans[0], interpolation='nearest', vmin=-m, vmax=m, cmap=plt.cm.RdBu_r)
    plt.title('FP')
    plt.subplot(212)
    plt.imshow(Gmeans[1], interpolation='nearest', vmin=-m, vmax=m, cmap=plt.cm.RdBu_r)
    plt.title('TP')
    plt.show()

if 0:
    Zmeans = [Z[i].mean(axis=-1) for i in xrange(2)]
    m = max(np.fabs(Zmeans[0]).max(), np.fabs(Zmeans[1]).max())

    plt.subplot(211)
    plt.imshow(Zmeans[0], interpolation='nearest', vmax=m)
    plt.title('FP')
    plt.subplot(212)
    plt.imshow(Zmeans[1], interpolation='nearest', vmax=m)
    plt.title('TP')
    plt.show()

# Open a shell where we can play around with the data
#import IPython
#IPython.embed()
