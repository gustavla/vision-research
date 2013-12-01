
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np


for i in xrange(1000):
    print i
    plt.figure()
    plt.imshow(np.random.randint(10, size=(100, 100)), interpolation='nearest')
    plt.savefig('db/{}.png'.format(i))
    plt.close()
    
