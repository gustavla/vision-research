from __future__ import division, print_function
import os
import argparse
import gv.datasets
import sys

def calc_scores(detector, filt, fn, positives=False):
    print('Processing file', fn, 'positives', positives)
    import os
    import textwrap
    import gv
    import amitgroup as ag
    import numpy as np

    detections = []
    img = gv.img.load_image(fn)
    grayscale_img = gv.img.asgray(img)
    grayscale_img = gv.imfilter.apply_filter(grayscale_img, filt)

    tp = tp_fp = tp_fn = 0

    scores, windows_count = detector.determine_scores(grayscale_img, one_centered=positives)

    return (scores, windows_count)

if gv.parallel.main(__name__):
    parser = argparse.ArgumentParser(description='Test response of model')
    parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')
    parser.add_argument('obj_class', metavar='<object class>', type=str, help='Object class')
    parser.add_argument('output', metavar='<output file>', type=argparse.FileType('wb'), help='Filename of output file')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--mini', action='store_true', default=False)
    parser.add_argument('--no-threading', action='store_true', default=False, help='Turn off threading')
    parser.add_argument('--size', type=int, default=None, help='Use fixed size')
    parser.add_argument('--param', type=float, default=None)
    parser.add_argument('--filter', type=str, default=None, help='Add filter to make detection harder')
    parser.add_argument('--classifier', action='store_true', default=False, help='Run as classifier and not detector')

    args = parser.parse_args()
    model_file = args.model
    obj_class = args.obj_class
    output_file = args.output
    limit = args.limit
    offset = args.offset
    threading = not args.no_threading

    import textwrap
    import gv
    import amitgroup as ag
    import numpy as np
    import os
    import itertools as itr
    import glob

    detector = gv.Detector.load(model_file)
    # TODO: New
    detector.TEMP_second = True
    detector._param = args.param

    if args.size is not None:
        detector.settings['min_size'] = detector.settings['max_size'] = args.size

    #dataset = ['val', 'train'][mini]
    #dataset = ['val', 'train'][mini]
    #dataset = 'val'

    print("Loading files...")
    #files, tot = gv.datasets.load_files(contest, obj_class)
    neg_files = sorted(glob.glob(os.path.expandvars('$INRIA_DIR/test_64x128_H96/neg/*')))
    pos_files = sorted(glob.glob(os.path.expandvars('$INRIA_DIR/test_64x128_H96/pos/*')))
    print("Done.")

    # Temporary
    if args.mini:
        pos_files = pos_files[:200]
        neg_files = neg_files[:40]

    FN = 0
    P = 0
    FP = 0
    N = 0

    detections = []

    all_kp_only_weights = None

    # Log the features and the model first

    # DO POSITIVES
    all_pos_scores = []
    argses = itr.izip(itr.repeat(detector), itr.repeat(args.filter), pos_files, itr.repeat(True))
    for scores, count in gv.parallel.starmap_unordered(calc_scores, argses):
        all_pos_scores.append(scores)
        P += count
    
    # DO NEGATIVES
    all_neg_scores = []
    argses = itr.izip(itr.repeat(detector), itr.repeat(args.filter), neg_files, itr.repeat(False))
    for scores, count in gv.parallel.starmap_unordered(calc_scores, argses):
        all_neg_scores.append(scores)
        N += count

    pscores = np.concatenate(all_pos_scores)
    nscores = np.concatenate(all_neg_scores) 


    ths = np.sort(pscores)
    miss_rates = np.zeros(ths.size) 
    fppws = np.zeros(ths.size)
    for i, th in enumerate(ths):
        TP = (pscores >= th).sum()
        FP = (nscores >= th).sum()

        miss_rates[i] = 1 - TP / P
        fppws[i] = FP / N

    # Also the reference threshold
    np.savez(output_file, miss_rates=miss_rates, fppws=fppws)

    # For each positive score, determine a point in the DET curve 

    #p, r = gv.rescalc.calc_precision_recall(detections, tot_tp_fn)
    #ap = gv.rescalc.calc_ap(p, r) 
    #np.savez(output_file, detections=detections, tp_fn=tot_tp_fn, tp_fn_dict=tp_fn_dict, ap=ap, contest=contest, obj_class=obj_class)

    ii = np.where(fppws > 1e-4)[0][-1]
    miss_rate_ref = miss_rates[ii]

    print('Miss rate at FPPW=1e-4: {}'.format(miss_rate_ref))

    print('FN', FN)
    print('FP', FP)
    print('P', P)
    print('N', N)
