from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='View mixture components')
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')

args = parser.parse_args()
model_file = args.model

import matplotlib.pylab as plt
import amitgroup as ag
import gv

# Load detector
detector = gv.Detector.load(model_file)

data = None
if detector.support is None:
    # Visualize feature activity if the support does not exist
    #assert 0, "This is broken since refactoring"
    data = detector.kernel_templates.sum(axis=-1)# / detector.kernel_templates.shape[-1] 
else:
    data = detector.support

#cag.plot.images(data, zero_to_one=True, caption=lambda i, im: "{0}: max: {1:.02} (w: {2:.02})".format(i, im.max(), detector.mixture.weights[i]))
#ag.plot.images(data, zero_to_one=True, caption=lambda i, im: "{0}: max: {1:.02} (w: {2:.02})".format(i, im.max(), 1.0))
ag.plot.images(data, zero_to_one=True, caption=lambda i, im: "")
