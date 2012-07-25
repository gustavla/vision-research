
import amitgroup as ag
import numpy as np


x = np.array([
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [1, 0, 0, 1]
    ])

mixture = ag.stats.BernoulliMixture(3, x)
mixture.run_EM(1e-4, save_template=True)

print mixture.templates

y = np.empty(x.shape[:-1])
for i, x_i in enumerate(x):
    y[i] = x_i[0] | (x_i[1] << 1) | (x_i[2] << 2) | (x_i[3] << 3)

mixture.save('mymix')

print y
mixture2 = ag.stats.BernoulliMixture(3, y)
mixture2.run_EM(1e-4, save_template=True)

print mixture2.templates
