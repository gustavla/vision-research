
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

ag.plot.images(detector.support, caption=lambda i, im: "{0}: max: {1:.02}".format(i, im.max()))
