
import amitgroup as ag
import gv
import numpy as np
from config import VOCSETTINGS

originals, bbs = gv.voc.load_object_images_of_size(VOCSETTINGS, 'bicycle', (64, 64), dataset='train')
#originals, bbs = gv.voc.load_negative_images_of_size(VOCSETTINGS, 'bicycle', (64, 64), dataset='train', count=1)

#print map(lambda x: x.shape, images)

#ag.plot.images(images[:9])

for i in xrange(4):
    import pylab as plt
    plt.imshow(originals[i], interpolation='nearest')

    from plotting import plot_box
    plot_box(bbs[i])
    plt.show()
