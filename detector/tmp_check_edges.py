from __future__ import division
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('settings', metavar='<settings file>', type=argparse.FileType('r'), help='Filename of settings file')

args = parser.parse_args()


import matplotlib as mpl
mpl.use('Agg')
import gv
import numpy as np
import amitgroup as ag
import glob
import matplotlib.pylab as plt
import os
from settings import load_settings
import itertools as itr

settings = load_settings(args.settings)
#d = gv.Detector.load('uiuc-supermodel01.npy')

parts_file = settings[settings['detector']['descriptor']]['file']
descriptor = gv.BinaryDescriptor.getclass('polarity-parts').load(parts_file)
esett = descriptor.bedges_settings()

#files = sorted(glob.glob(os.path.expandvars('$UIUC_DIR/TestImages/*.pgm')))
files = sorted(glob.glob(os.path.expandvars('$VOC_DIR/JPEGImages/*.jpg')))

#files = [np.random.uniform(size=(200, 200))]

#esett['minimum_contrast'] = 0.2

all_ors = []

for count, f in enumerate(files[:25]):
    if isinstance(f, np.ndarray):
        im = f
    elif 0:
        # Random noise, for testing
        im = np.random.uniform(0, 1, size=(400, 400))
    else:
        #im = gv.img.asgray(gv.img.load_image(f))
        from PIL import Image
        im = Image.open(f)
        im = im.rotate(np.random.randint(360))
        im = gv.img.asgray(np.array(im).astype(np.float64)/255.0)

        c = np.min(im.shape)

        c1 = c / np.sqrt(2)

    # TODO:

    if 0:
        from gv.gradients import orientations
        ors, gr_x, gr_y = orientations(im, orientations=8)

        if ors.size == 0:
            continue

        all_ors.append(ors.ravel())

        hh, xedges, yedges = np.histogram2d(gr_y.ravel(), gr_x.ravel(), bins=50)
        plt.figure()
        #plt.hist(ors.ravel(), 36*2)
        plt.imshow(hh, interpolation='nearest', origin='low',
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
        plt.savefig('edges2/{}-hist2d.png'.format(count))
        plt.close()

        plt.figure()
        plt.hist(ors.ravel(), 36*2)
        plt.savefig('edges2/{}-hist.png'.format(count))
        plt.close()

    gv.img.save_image('edges2/{}-aimage.png'.format(count), im)

    #edges = ag.features.bedges(im, **esett)
    edges = descriptor._extract_edges(im)

    edges = ag.features.bspread(edges, radius=settings['edges']['radius'], spread=settings['edges']['spread'])

    plt.clf()
    fig, axarr = plt.subplots(2, 2)#, figsize=(23, 15))

    cmap = mpl.colors.ListedColormap(['white', 'blue', 'red', 'yellow'])
    EDGE_TITLES = ['- Horizontal', '/ Diagonal', '| Vertical', '\\ Diagonal']

    for i in xrange(4):
        ax = axarr.ravel()[i]
        ax.set_axis_off()
        ax.imshow(edges[...,i] + 2*edges[...,i+4], cmap=cmap, vmin=0, vmax=3, interpolation='nearest')
        ax.set_title(EDGE_TITLES[i])

    fig.savefig('edges2/{}-edges.png'.format(count))
    plt.close()

    # Now, extract parts

    if 0:
        #partprobs = descriptor.extract_partprobs(im)
        feats = descriptor.extract_features(im, settings=dict(spread_radii=(0, 0), subsample_size=(1, 1)))
        
        import scipy
        #mus = np.concatenate(([0], scipy.ndimage.zoom(descriptor.extra['means_emp'], 2, order=0)))
        #sigmas = np.concatenate(([1], scipy.ndimage.zoom(descriptor.extra['stds_emp'], 2, order=0)))
        mus = np.concatenate(([0], descriptor.extra['means_emp']))
        sigmas = np.concatenate(([1], descriptor.extra['stds_emp']))

        #argmax_partprobs = partprobs.argmax(axis=-1)
        #import pdb; pdb.set_trace()
        #partprobs = (partprobs - mus) / sigmas

        max_partprobs = np.zeros(feats.shape[:2])

    if 0:
        sh = descriptor.parts.shape[1:3]
        for i, j in itr.product(xrange(feats.shape[0]), xrange(feats.shape[1])):
            #max_partprobs[i,j] = partprobs[i,j,argmax_partprobs[i,j]]
            ii = feats[i,j].argmax()
            index = ii

            #if ii == 0:
                #max_partprobs[i,j] = 0 
            #else:
            if feats[i,j,ii] == 0:
                max_partprobs[i,j] = -1
            else:
                obj = descriptor.parts[index]
                #avg = np.tile(obj.mean(axis=-1)[...,np.newaxis], obj.shape[-1])
                avg = np.tile(np.apply_over_axes(np.mean, obj, [0, 1]), (obj.shape[0], obj.shape[1], 1))
                X = edges[i:i+sh[0],j:j+sh[1]]
                wplus = np.log(obj / avg)
                wminus = np.log((1 - obj) / (1 - avg))

                v = np.sum(X * wplus) + np.sum((1 - X) * wminus)


                
                #if ii == 0:
                    #v = np.nan
                #else:
                    #v = mus[ii]

                #bkg = descriptor.parts


                #max_partprobs[i,j] = v
                #max_partprobs[i,j
                #max_partprobs[i,j] = partprobs[i,j,ii]# mus[ii] / sigmas[ii]# partprobs[i,j,ii]  #mus[ii]# / sigmas[ii]
                max_partprobs[i,j] = hash(ii**20)%200#partprobs[i,j,ii]# mus[ii] / sigmas[ii]# partprobs[i,j,ii]  #mus[ii]# / sigmas[ii]

    #print partprobs.shape
    #max_partprobs = partprobs.max(axis=-1)
    #import scipy.stats
    #max2 = scipy.stats.scoreatpercentile(partprobs, 98, axis=-1)
    #print max2.shape

    #max_partprobs[max_partprobs == 0] = np.nan

    if 0:
        fig = plt.figure()
        plt.subplot(111)
        mm = max(max_partprobs.max(), -max_partprobs.min())
        print mm
        plt.imshow(max_partprobs, interpolation='nearest')#, vmin=-mm, vmax=mm, cmap=plt.cm.RdBu_r)
        plt.colorbar()
        plt.savefig('edges2/{}-partprobs.png'.format(count))
        plt.close()


        #fig = plt.figure()
        #plt.subplot(111)
        #plt.imshow(max2, interpolation='nearest')
        #plt.colorbar()
        #plt.savefig('edges/{}-max2.png'.format(count))
        #plt.close()

        
        feats = descriptor.extract_features(im)
        # Feature density
        print 'Feature density:', feats.mean()

if 0:
    plt.figure()
    plt.hist(np.concatenate(all_ors, axis=0), 36*4)
    plt.savefig('edges2/aaaall-hist.png'.format(count))
    plt.close()

    plt.figure()
    plt.hist(np.concatenate(all_ors, axis=0), 360)
    plt.savefig('edges2/aaaall-hist360.png'.format(count))
    plt.close()

    plt.figure()
    plt.hist(np.concatenate(all_ors, axis=0), 8)
    plt.savefig('edges2/aaaall-hist8.png'.format(count))
    plt.close()

    plt.figure()
    plt.hist(np.concatenate(all_ors, axis=0), 18)
    plt.savefig('edges2/aaaall-hist18.png'.format(count))
    plt.close()
