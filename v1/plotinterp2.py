
import numpy as np
import amitgroup as ag
import matplotlib.pylab as plt

face = ag.io.load_example('faces')[2]

#face = np.array([[0.0, 1.0], [0.00, 1.00]])

#face = face[::-1]

xs = np.empty((face.shape[0]*20, face.shape[1]*20, 2))
for x0 in range(xs.shape[0]):
    for x1 in range(xs.shape[1]):
        xs[x0,x1] = np.array([x0/float(xs.shape[0]), x1/float(xs.shape[1])])

face2 = ag.math.interp2d(xs, face, dx=np.array([1.0/(face.shape[0]-1), 1.0/(face.shape[1]-1)]))

d = dict(cmap=plt.cm.gray, interpolation='nearest')
plt.figure(figsize=(14,8))
plt.subplot(121)
plt.imshow(face, **d)
plt.subplot(122)
plt.imshow(face2, **d)
plt.show()
