from __future__ import print_function
import numpy as np
import amitgroup as ag
import amitgroup.stats
import amitgroup.features
import matplotlib.pylab as plt
import sys

# generate synthetic data

XS_ = np.load("car.npy")

edges = ag.features.bedges(XS_, 5)
print(edges.shape)
edges = np.rollaxis(edges, 3)
print(edges.shape)
XS = edges[int(sys.argv[1])]

sh = 3, 3
num_mixtures = np.prod(sh)

bm = ag.stats.BernoulliMixture(num_mixtures, XS)
bm.run_EM(.0001)

np.savez('output.npz', weights=bm.weights, templates=bm.templates)

plt.figure(figsize=(8,8))
for i in range(num_mixtures)[:np.prod(sh)]:
    plt.subplot(sh[0], sh[1], 1+i).set_axis_off()
    plt.imshow(bm.templates[i], interpolation='nearest', cmap=plt.cm.gray)

print(XS)
print(bm.weights)

plt.show()
