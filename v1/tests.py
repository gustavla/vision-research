
import numpy as np
import matplotlib.pylab as plt


images = np.load('nines.npz')['images']
im = images[0]
im = im[::-1,:]

plt.figure(figsize=(14,6))
plt.subplot(121)
plt.xlabel('x')
plt.ylabel('y')
plt.imshow(im, interpolation='nearest', origin='lower')

plt.subplot(122)
plt.quiver(im, np.zeros(im.shape))

plt.show()

