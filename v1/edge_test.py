

import numpy as np
import amitgroup as ag
import matplotlib.pylab as plt

#digit = ag.io.load_example('mnist')[1]
digits, _ = ag.io.load_mnist('testing')
digit = digits[1]


#edges = ag.features.bedges(digit, k=4, inflate=False)
edges2 = ag.features.bedges(digit, k=4)

#ag.plot.images(np.c_[np.rollaxis(edges, 2), np.rollaxis(edges2, 2)])
ag.plot.images(np.rollaxis(edges2, 2))
plt.show()
