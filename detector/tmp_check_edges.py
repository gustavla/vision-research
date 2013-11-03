
import matplotlib as mpl
mpl.use('Agg')
import gv
import numpy as np
import amitgroup as ag
import glob
import matplotlib.pylab as plt
import os

d = gv.Detector.load('uiuc-supermodel01.npy')


esett = d.descriptor.bedges_settings()

files = sorted(glob.glob(os.path.expandvars('$UIUC_DIR/TestImages/*.pgm')))

#files = [np.random.uniform(size=(200, 200))]

#esett['minimum_contrast'] = 0.2

for count, f in enumerate(files[:10]):
    if isinstance(f, np.ndarray):
        im = f
    else:
        im = gv.img.load_image(f)

    gv.img.save_image('edges/{}-aimage.png'.format(count), im)

    edges = ag.features.bedges(im, **esett)

    plt.clf()
    fig, axarr = plt.subplots(2, 2)#, figsize=(23, 15))

    cmap = mpl.colors.ListedColormap(['white', 'blue', 'red', 'yellow'])
    EDGE_TITLES = ['- Horizontal', '/ Diagonal', '| Vertical', '\\ Diagonal']

    for i in xrange(4):
        ax = axarr.ravel()[i]
        ax.set_axis_off()
        ax.imshow(edges[...,i] + 2*edges[...,i+4], cmap=cmap, vmin=0, vmax=3, interpolation='nearest')
        ax.set_title(EDGE_TITLES[i])

    fig.savefig('edges/{}-edges.png'.format(count))
    plt.close()

    # Now, extract parts

    partprobs = d.descriptor.extract_partprobs(im)
    #print partprobs.shape
    max_partprobs = partprobs.max(axis=-1)
    import scipy.stats
    max2 = scipy.stats.scoreatpercentile(partprobs, 95, axis=-1)
    print max2.shape

    max_partprobs[max_partprobs == 0] = np.nan

    fig = plt.figure()
    plt.subplot(111)
    plt.imshow(max_partprobs, interpolation='nearest')
    plt.colorbar()
    plt.savefig('edges/{}-partprobs.png'.format(count))
    plt.close()


    fig = plt.figure()
    plt.subplot(111)
    plt.imshow(max2, interpolation='nearest')
    plt.colorbar()
    plt.savefig('edges/{}-max2.png'.format(count))
    plt.close()

    
    feats = d.descriptor.extract_features(im)
    # Feature density
    print 'Feature density:', feats.mean()

