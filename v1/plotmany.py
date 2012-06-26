import amitgroup as ag
import amitgroup.io
import matplotlib.pylab as plt
import sys

images = ag.io.load_all_images('/local/base/book/data/FACES/register_32/Images_1')

index = int(sys.argv[1])

plt.figure()
for i in range(9):
    plt.subplot(331+i)
    plt.imshow(images[index+i], cmap=plt.cm.gray, interpolation='nearest')
plt.show()
