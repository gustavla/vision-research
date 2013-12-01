from __future__ import division
from __future__ import print_function
import argparse

parser = argparse.ArgumentParser(description='Test response of model')
parser.add_argument('results', metavar='<file>', nargs='+', type=argparse.FileType('rb'), help='Filename of results file')
parser.add_argument('--captions', metavar='<caption>', nargs='*', type=str, help='Captions')

args = parser.parse_args()
results_files = args.results
captions = args.captions
if captions:
    assert len(captions) == len(results_files), "Must supply caption for all"

import numpy as np
import matplotlib.pylab as plt
import gv

use_other_style = False 


for i, results_file in enumerate(results_files):
    data = np.load(results_file)

    detections = data['detections']
    tp_fn_dict = data['tp_fn_dict'].flat[0]
    N = len(detections)

    tp_fn = int(data['tp_fn'])
    p, r = gv.rescalc.calc_precision_recall(detections, tp_fn)
    ap = gv.rescalc.calc_ap(p, r) 

    print(results_file.name)
    print('AP: {0:.02f}% ({1})'.format(100*ap, ap))
    #print(detections[-10:])
    print()



    # Split detections into files
    img_ids = {det['img_id'] for det in detections} 
    #img_ids_list = list(img_ids)
    id_to_i = {img_id: i for i, img_id in enumerate(img_ids)}
    i_to_id = {i: img_id for i, img_id in enumerate(img_ids)}

    hierdets = {img_id: detections[detections['img_id']==img_id] for img_id in img_ids}

    M = len(img_ids)

    # Bootstrap

    TRIALS = 1000
    aps = np.zeros(TRIALS) 
    for loop in xrange(TRIALS):
        rs = np.random.RandomState(loop)
        II = rs.randint(M, size=M)
        det_boot = np.concatenate([hierdets[i_to_id[ii]] for ii in II])
        tp_fn_boot = np.sum([tp_fn_dict[i_to_id[ii]] for ii in II])

        det_boot.sort(order='confidence')
        
        #det_boot = detections[II]
    
        #print(tp_fn_boot)
        if tp_fn_boot == 0:
            ap = 1.0
        else:
            p, r = gv.rescalc.calc_precision_recall(det_boot, tp_fn_boot)
            ap = gv.rescalc.calc_ap(p, r) 
        aps[loop] = ap
        #print('AP: {0:.02f}% ({1})'.format(100*ap, ap))

    print('-'*30)
    print('Average AP: {0:.02f}%'.format(100*np.mean(aps)))
    print('Standard error: {0:.02f}%'.format(100*np.std(aps, ddof=1)))
    print('{0:.02f} +- {1:.02f}'.format(100*np.mean(aps), 100*1.96*np.std(aps, ddof=1)))
