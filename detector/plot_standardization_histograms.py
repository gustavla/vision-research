
import numpy as np
from pylab import *

all_llhs = [np.load('llhs-{0}.npy'.format(i)) for i in xrange(6)]

ra = (-1000, 8000)
ra = (-2000, 5000)

for i, llhs in enumerate(all_llhs):
    subplot(3, 2, 1+i)
    hist(llhs, normed=True, bins=np.arange(ra[0], ra[1], 250))
    xlim(ra)
    mu, sigma = np.mean(llhs), np.std(llhs)
    print mu, sigma
    x = np.linspace(ra[0], ra[1], 100)
    y = np.exp(-(x - mu)**2 / (2*sigma**2)) / np.sqrt(2*np.pi*sigma**2)
    plot(x, y, color='red')

show()
