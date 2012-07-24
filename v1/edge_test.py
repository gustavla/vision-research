

import numpy as np
import amitgroup as ag
import matplotlib.pylab as plt

digit = ag.io.load_example('mnist')[0]


edges = ag.features.bedges(digit, inflate=False)
edges2 = ag.features.bedges(digit)

ag.plot.images(np.c_[np.rollaxis(edges, 2), np.rollaxis(edges2, 2)])
plt.show()
