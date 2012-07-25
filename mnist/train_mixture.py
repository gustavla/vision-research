
from __future__ import print_function
import amitgroup as ag
from amitgroup.stats import BernoulliMixture
import numpy as np
import sys


try:
    filename = sys.argv[1]
except IndexError:
    print("boo boo, need filename!")
    sys.exit(0)

data = np.load(filename)

def train_mixture(data):
    for d in range(10):
        print(d)
        str_d = str(d)
        digits = data[str_d]
        #digits = digits[:50]

        # Train a mixture model
        mixture = BernoulliMixture(9, digits)
        mixture.run_EM(1e-3)
        
        mixture.save('mix/mixture-digit-{0}'.format(d))
        break

import cProfile as profile
if __name__ == '__main__':
    profile.run('train_mixture(data)')
    #train_mixture(data)
