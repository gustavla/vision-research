from __future__ import division
import numpy as np
#import matplotlib.pylab as plt
import gv

from imgdata import getImageInfo

scales = []

objects = ['bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
#objects = ['car']

min_size = np.inf

min_sizes = []

for obj in objects:
    files, _ = gv.datasets.load_files('voc-trainval', obj) 
    for f in files:
        # Check size
        with open(f.path, 'rb') as fp:
            data = fp.read(300)

        _, w, h = getImageInfo(data)
        area = w * h
        min_sizes.append(min(w, h))

        if min(w, h) < 100:
            print f.path, w, h
        min_size = min(min_size, w, h)
         
        for bb in f.boxes:
            #scale = np.sqrt(gv.bb.area(bb.box)) / np.sqrt(area)
            if not bb.truncated:
                scale = np.max(gv.bb.size(bb.box))# / max(w, h)
                #print scale, area
                scales.append(scale)

min_sizes = np.asarray(min_sizes)
scales = np.asarray(scales)

print 'min_size', min_size
np.save('min_sizes.npy', min_sizes)
np.save('scales2.npy', scales)

# 0.50: < 13
# 0.75: 13.45
# 1.50: 17.31
# 1.75: 17.48
# 2.00: 17.31 

#scales = np.random.uniform(10, 500, size=10000)

lscales = np.log(scales/100) / np.log(2)

if 0:
    ax = plt.subplot(111)
    ax.hist(lscales, weights=scales**2, normed=True, bins=30)
    plt.xlabel('image pyramid level (log2 of scale factor)')
    plt.ylabel('Object density (normalized scale)')
    ax.set_xscale('log')
    #ax.set_yscale('log')

    #plt.hist(scales, normed=True)
    plt.show()
