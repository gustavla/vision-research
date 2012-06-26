import amitgroup as ag
import amitgroup.io
import matplotlib.pylab as plt
import sys

image0 = ag.io.load_image('/local/base/book/data/FACES/register_32/Images_0', 45)
image1 = ag.io.load_image('/local/base/book/data/FACES/register_32/Images_1', 23)

images = image0, image1

for i in range(2):
    plt.subplot(121+i)
    plt.imshow(images[i], cmap=plt.cm.gray, interpolation='nearest')
plt.show()
