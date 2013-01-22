

import gv

img = gv.img.load_image('brick.jpg')


img2 = img.copy()
#for s in [256, 128, 64]:
    #img2 = gv.img.resize(img2, (s, s))

img2 = gv.img.resize(img, (1024, 1024))
img3 = gv.img.resize(img, (1024, 1024), prefilter=False)

import matplotlib.pylab as plt

settings = dict(interpolation='nearest')

fig = plt.figure(figsize=(15, 5))

if 0:
    plt.subplot(131)
    plt.imshow(img, **settings)
    plt.title('Full size')

    plt.subplot(132)
    plt.imshow(img2, **settings)
    plt.title('Subsampled')

    plt.subplot(133)
    plt.imshow(img3, **settings)
    plt.title('Not pre-filtered')
    plt.show()
else:
    gv.img.save_image(img2, 'output.png')

