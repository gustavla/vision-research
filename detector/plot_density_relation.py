
from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='View mixture components')
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')
parser.add_argument('img_id', metavar='<image id>', type=int, help='ID of image in VOC repository')
parser.add_argument('mixcomp', metavar='<mixture component>', type=int, help='mix comp')

args = parser.parse_args()
model_file = args.model
img_id = args.img_id
mixcomp = args.mixcomp

import gv
import matplotlib.pylab as plt
from config import VOCSETTINGS
import numpy as np

# Load detector
detector = gv.Detector.load(model_file)

fileobj = gv.voc.load_training_file(VOCSETTINGS, 'bicycle', img_id)
img = gv.img.load_image(fileobj.path)

# Run detection (mostly to get resized image)
back, kernels, _ = detector.prepare_kernels(img, mixcomp)

densities = np.linspace(0, 1 + 1e-10, 50)

sh = kernels.shape[1:]

scores = np.empty(len(densities))
for i, density in enumerate(densities):
    window = (np.random.random(sh) < density).astype(np.float64)
    #print "Density", density, window.sum()/np.prod(window.shape)

    #contribution_map = np.zeros(sh[:-1]) 
    data = \
        window * np.log(kernels[mixcomp]) + \
        (1.0-window) * np.log(1.0 - kernels[mixcomp]) + \
        (-1) * (1.0-window) * np.log(1.0 - back[0,0]) + \
        (-1) * window * np.log(back[0,0])
    #contribution_map += data

    scores[i] = data.sum()#contribution_map.sum()
    

plt.plot(densities, scores)
plt.xlim((0, 1))
plt.xlabel('Feature density')
plt.ylabel('Log likelihood ratio')
plt.title('Scores for random feature activity')
plt.show()

