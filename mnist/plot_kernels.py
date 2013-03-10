
import numpy as np
import amitgroup as ag
import matplotlib.pylab as plt
zeropad = ag.util.zeropad

radius = 4
kernels = [
    ("box{0}".format(i), zeropad(np.ones((1+2*i, 1+2*i)), radius-i)) for i in range(0, 4)
] + [
    ("orthogonal{0}".format(i), zeropad(np.eye(1+2*i), radius-i)) for i in range(1, 4)
]

for i, (name, kernel) in enumerate(kernels):
    print i
    plt.subplot(331 + i)
    plt.xlabel(name)
    plt.imshow(kernel, cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
