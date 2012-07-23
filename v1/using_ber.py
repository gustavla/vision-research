from __future__ import print_function
import numpy as np
import amitgroup as ag
import amitgroup.stats
import matplotlib.pylab as plt

# generate synthetic data

XS = np.load("test.npz")['data']

num_mixtures = 18 

bm = ag.stats.BernoulliMixture(num_mixtures, XS)
bm.run_EM(.0001)

np.savez('output.npz', weights=bm.weights, templates=bm.templates)

plt.figure(figsize=(14,5))
for i in range(num_mixtures)[:3*9]:
    plt.subplot(3, 6, 1+i)
    plt.imshow(bm.templates[i], interpolation='nearest', cmap=plt.cm.gray)

print(XS)
print(bm.weights)

plt.show()
