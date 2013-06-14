from __future__ import division
import argparse

parser = argparse.ArgumentParser(description='Test response of model')
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')
parser.add_argument('mixcomp', metavar='<mixture component>', type=int, help='mix comp')
parser.add_argument('image', metavar='<image file>', type=argparse.FileType('rb'), help='Filename of image file')
parser.add_argument('output', metavar='<output file>', type=argparse.FileType('wb'), help='Filename of output file')

args = parser.parse_args()
model_file = args.model
mixcomp = args.mixcomp
image_file = args.image
output_file = args.output

import matplotlib
matplotlib.use('Agg')
import numpy as np
import gv
import matplotlib.pylab as plt

detector = gv.Detector.load(model_file)

orig_im = gv.img.asgray(gv.img.load_image(image_file))

psize = detector.settings['subsample_size']
radii = detector.settings['spread_radii']
cb = detector.settings.get('crop_border')

sub_kernels = detector.prepare_kernels(None, settings=dict(spread_radii=radii, subsample_size=psize))

values = []
resmaps = []
factors = np.arange(1.5, 0.5, -0.01)
mn, mx = np.inf, -np.inf

for factor in factors:
    im = gv.img.resize_with_factor_new(orig_im, factor)

    spread_feats = detector.descriptor.extract_features(im, dict(spread_radii=radii, preserve_size=False))
    sub_feats = gv.sub.subsample(spread_feats, psize) 
    spread_bkg = detector.bkg_model(None, spread=True)
    resmap = detector.response_map(sub_feats, sub_kernels, spread_bkg, mixcomp, level=-1)

    index = np.argmax(resmap) 
    pos = np.unravel_index(index, resmap.shape)
    print 'size', im.shape 
    print 'pos',  np.array(pos) / factor
    print 'value', resmap[pos]
    values.append(resmap[pos])

    mn = min(mn, resmap.min())
    mx = max(mx, resmap.max())

    resmaps.append(resmap)

    if 0:
        plt.clf()
        plt.imshow(resmap, interpolation='nearest')
        plt.colorbar()
        plt.savefig('doutput/resmap-factor-{0:.2f}.png'.format(factor)) 

    if 0:
        plt.clf()
        plt.imshow(im, interpolation='nearest', cmap=plt.cm.gray)
        plt.savefig('doutput/image-factor-{0:.2f}.png'.format(factor))

    #= gv.img.resize_with_factor_new(img, 1/factor) 

    #bbs, x, feats, img_resized = detector.detect_coarse_single_factor(im, factor, mixcomp)

for i, factor in enumerate(factors):
    resmap = resmaps[i] 
    plt.clf()
    plt.imshow(resmap, interpolation='nearest', vmin=mn, vmax=mx)
    plt.colorbar()
    plt.savefig('doutput/resmap-factor-{0:.2f}.png'.format(factor)) 
    
plt.clf()
plt.plot(factors, values)
plt.savefig('doutput/histogram.png')
