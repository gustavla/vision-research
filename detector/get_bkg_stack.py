from __future__ import division
import glob
import numpy as np
import amitgroup as ag
import gv
import os
import sys
from itertools import product
from superimpose_experiment import generate_random_patches

def get_bkg_stack(settings, X_pad_size, M=20):
    descriptor = gv.load_descriptor(settings)

    bsettings = settings['edges'].copy()
    radius = bsettings['radius']
    bsettings['radius'] = 0

    neg_filenames= sorted(glob.glob(os.path.join(os.environ['UIUC_DIR'], 'TrainImages', 'neg-*.pgm')))

    gen_raw = generate_random_patches(neg_filenames, X_pad_size, 0, per_image=25)

    bkg_stack_num = np.zeros(descriptor.num_parts + 1)
    bkg_stack = np.zeros((descriptor.num_parts + 1, M,) + X_pad_size)

    i = 0
    import matplotlib.pylab as plt
    N = 100000
    for patch in gen_raw:
        edges = ag.features.bedges(patch, **bsettings)

        #plt.imshow(patch, interpolation='nearest', cmap=plt.cm.gray)
        #plt.show()

        X_pad_spread = ag.features.bspread(edges, spread=bsettings['spread'], radius=radius)

        padding = pad - 2
        X_spread = X_pad_spread[padding:-padding,padding:-padding]

        # Code parts 
        parts = descriptor.extract_parts(X_spread.astype(np.uint8))

        # Accumulate and return
        if parts[0,0].sum() == 0:
            f = 0 
        else:
            f = np.argmax(parts[0,0]) + 1
            #cc[f] += 1

        # The i%10 is to avoid all background images for f=0 to be from the same image (and thus
        # likely overlapping patches)
        if bkg_stack_num[f] < M and (f != 0 or i%10 == 0):
            bkg_stack[f,bkg_stack_num[f]] = patch
            bkg_stack_num[f] += 1

        if i % 10000 == 0:
            print i, bkg_stack_num
        i += 1
        if i == N: 
            break

    #print 'i', i

    #print 'min', sorted(cc)[:10] 
    #cc /= N
    #print cc[:10]
    #print bkg[:10]

    #print cc.sum()
    #print bkg.sum()
    return bkg_stack, bkg_stack_num

if __name__ == '__main__':
    import argparse
    from settings import load_settings
   
    ag.set_verbose(True)
    
    parser = argparse.ArgumentParser(description="Convert model to integrate background model")
    parser.add_argument('settings', metavar='<settings file>', type=argparse.FileType('r'), help='Filename of settings file')
    parser.add_argument('bkgstack', metavar='<bkgstack file>', type=argparse.FileType('wb'), help='Bkg stack output file')
    args = parser.parse_args()
    settings_file = args.settings
    output_file = args.bkgstack

    settings = load_settings(settings_file)

    pad = 5 
    size = settings['parts']['part_size'] 
    X_pad_size = (size[0]+pad*2, size[1]+pad*2)

    bkg_stack, bkg_stack_num = get_bkg_stack(settings, X_pad_size, M=20)

    np.savez(output_file, bkg_stack=bkg_stack, bkg_stack_num=bkg_stack_num)

