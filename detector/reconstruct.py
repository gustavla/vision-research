from __future__ import print_function
from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='Reconstruct from parts-based object model')
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')
parser.add_argument('mixcomp', metavar='<mixture component>', type=int, help='mix comp')

args = parser.parse_args()
model_file = args.model
mixcomp = args.mixcomp

import gv
import amitgroup as ag
import numpy as np
import matplotlib.pylab as plt

detector = gv.Detector.load(model_file)

kerns = detector.kernel_templates

r = np.zeros(kerns.shape[1:3])

sh = detector.descriptor.visparts.shape[-2:]
pad = (20, 20)

r = ag.util.zeropad(r, pad)
r2 = np.zeros(r.shape)

#import pdb; pdb.set_trace()
for i in xrange(0, kerns.shape[1], 1):
    for j in xrange(0, kerns.shape[2], 1):
        part = np.argmax(kerns[mixcomp,i,j])
        print(part)
        
        if kerns[mixcomp,i,j,part] > 0.01:
            print(pad[0]+i-sh[0]//2, pad[0]+i-sh[0]//2 + sh[0],\
              pad[1]+j-sh[1]//2, pad[1]+j-sh[1]//2 + sh[1])
            r[pad[0]+i-sh[0]//2: pad[0]+i-sh[0]//2 + sh[0],\
              pad[1]+j-sh[1]//2: pad[1]+j-sh[1]//2 + sh[1]] += detector.descriptor.visparts[part]

            r2[pad[0]+i-sh[0]//2: pad[0]+i-sh[0]//2 + sh[0],\
              pad[1]+j-sh[1]//2: pad[1]+j-sh[1]//2 + sh[1]] += 1 

#plt.imshow(kerns[mixcomp].sum(axis=-1), interpolation='nearest')
plt.imshow(r/r2, interpolation='nearest', cmap=plt.cm.gray)
plt.show()
