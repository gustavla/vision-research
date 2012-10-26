
import argparse

parser = argparse.ArgumentParser(description='Train mixture model on edge data')
parser.add_argument('patch', metavar='<patch file>', type=argparse.FileType('rb'), help='Filename of patches file')
parser.add_argument('image', metavar='<image file>', type=argparse.FileType('rb'), help='Filename of image file to detect in')

args = parser.parse_args()
patch_file = args.patch
image_file = args.image

import numpy as np
import matplotlib.pylab as plt
import amitgroup as ag
import amitgroup.features

patch_data = np.load(patch_file)
patches = patch_data['patches']

edges, img = ag.features.bedges_from_image(image_file, k=5, radius=1, minimum_contrast=0.05, return_original=True, lastaxis=True)

# Now, pre-process the log parts
log_parts = np.log(patches)
log_invparts = np.log(1-patches)

threshold = 25 

print edges.shape
print log_parts.shape

#ret3 = ag.features.spread_patches(edges, 5, 5, 25)
#print ret3.shape

ret = ag.features.code_parts(edges, log_parts, log_invparts, threshold)

#print ret.shape

#print ret[32, 32]
#print ret[0, 0]

ret2 = ret.argmax(axis=2)

#print ret2, ret2.min(), ret2.max()

plt.subplot(121)
plt.imshow(img, interpolation='nearest')
plt.subplot(122)
plt.imshow(ret2, interpolation='nearest')
plt.colorbar()
plt.show()
#plt.imshow(
