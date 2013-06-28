from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='Test response of model')
#parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')
parser.add_argument('results', metavar='<file>', nargs=1, type=argparse.FileType('rb'), help='Filename of results file')
parser.add_argument('output_dir', metavar='<output folder>', nargs=1, type=str, help="Output path")
parser.add_argument('--count', metavar='COUNT', nargs=1, default=[20], type=int, help='The k in top-k')
#parser.add_argument('img_id', metavar='<image id>', type=int, help='ID of image in VOC repository')

args = parser.parse_args()
#model_file = args.model
results_file = args.results[0]
output_dir = args.output_dir[0]
count = args.count[0]

#img_id = args.img_id

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import gv
import numpy as np
from plotting import plot_image
import os

#detector = gv.Detector.load(model_file)
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

for i, det in enumerate(detections[:count]):
    bb = (det['top'], det['left'], det['bottom'], det['right'])
    bbobj = gv.bb.DetectionBB(bb, score=det['confidence'], confidence=det['confidence'], mixcomp=det['mixcomp'], correct=det['correct'])

    img_id = det['img_id']
    fileobj = gv.datasets.load_file(contest, img_id, obj_class=obj_class)
    
    # Replace bounding boxes with this single one
    fileobj.boxes[:] = [bbobj]
    
    fn = 'det-{0}.png'.format(i)
    path = os.path.join(output_dir, fn)
    plot_image(fileobj, filename=path, show_corrects=True)
    print 'Saved {0}'.format(path)
