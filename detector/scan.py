
import argparse

parser = argparse.ArgumentParser(description='Test response of model')
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')
parser.add_argument('img', metavar='<image file>', type=argparse.FileType('rb'), help='Filename of image file')

args = parser.parse_args()
model_file = args.model
image_file = args.img

import gv
import numpy as np
from PIL import Image

detector = gv.Detector.load(model_file)

img = np.array(Image.open(image_file)).astype(np.float64) / 255.0

x, small = detector.response_map(img)

# Get max peak
ix, iy = np.unravel_index(x.argmax(), x.shape)
ix *= detector.patch_dict.settings['pooling_size'][0]
iy *= detector.patch_dict.settings['pooling_size'][1]
#print ix, iy

import matplotlib.pylab as plt

#print '---'
#print x.shape
#print small.shape

plt.subplot(221)
plt.title('Input image')
#print detector.small_support.shape
#plt.imshow(detector.small_support[2], interpolation='nearest')
plt.imshow(img)
#w, h = [detector.mixture.templates.shape[1+i]//2 * detector.patch_dict.settings['pooling_size'][i] for i in range(2)]

supp_size = detector.support[2].shape
bb = detector.get_support_box_for_mix_comp(2)

#print 'bb info'
#print supp_size
#print bb

#plt.gca().add_patch(plt.Rectangle((iy-h, ix-w), 2*h, 2*w, facecolor='none'))
plt.gca().add_patch(plt.Rectangle((iy-supp_size[0]//2+bb[0][0], ix-supp_size[1]//2+bb[0][1]), bb[1][0]-bb[0][0], bb[1][1]-bb[0][1], facecolor='none', edgecolor='cyan', linewidth=2.0))
plt.colorbar()

plt.subplot(222)
plt.title('Response map')
plt.imshow(x, interpolation='nearest')
plt.colorbar()

plt.subplot(223)
plt.title('Feature activity')
plt.imshow(small.sum(axis=-1), interpolation='nearest')
plt.colorbar()

plt.subplot(224)
if 0:
    pass
    plt.title('Normalized stuff')
    plt.imshow(x / np.clip(small.sum(axis=-1), 5, np.inf), interpolation='nearest')
    plt.colorbar()
else:
    plt.title('Kernel Bernoulli probability averages')
    plt.imshow(detector.kernels[2].mean(axis=-1), interpolation='nearest', cmap=plt.cm.RdBu, vmin=0, vmax=1)
    plt.colorbar()


plt.show()

#np.save('x', x)
