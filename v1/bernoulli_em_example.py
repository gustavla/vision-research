from __future__ import print_function
import numpy as np
import amitgroup as ag
import amitgroup.stats

# generate synthetic data
num_templates = 3
template_size = (6,6)
templates = np.zeros((num_templates,
                      template_size[0],
                      template_size[1]))
for T in templates:
    x = np.random.randint(template_size[0]-4)
    y = np.random.randint(template_size[1]-4)
    T[x:x+4,
      y:y+4] = .95
    T = np.maximum(.05,T)

for T in templates:
    T = np.maximum(.05,T)


num_data = 100
XS = np.zeros((num_data,template_size[0],template_size[1]))

for X in XS:
    S = np.random.rand(template_size[0],template_size[1])
    X[S < templates[np.random.randint(num_templates)]] = 1.


bm = ag.stats.BernoulliMixture(2,XS)
bm.run_EM(.0001)

print(XS)
print(XS.shape)
print(bm.weights)
print(bm.templates)
print(bm.templates.shape)

