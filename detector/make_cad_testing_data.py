from __future__ import division, print_function
import argparse

parser = argparse.ArgumentParser(description='Generate testing data')
parser.add_argument('name', type=str, help='Name of dataset')
parser.add_argument('--limit', type=int, default=20)
parser.add_argument('--perimage', type=int, default=1)

args = parser.parse_args()
name = args.name
limit = args.limit
perimage = args.perimage

import random
import os
import gv
import glob
import itertools
import numpy as np

random.seed(0)
#neg_files = sorted(glob.glob(os.path.expandvars('$VOC_DIR/JPEGImages/*.jpg')))
neg_files = sorted(glob.glob(os.path.expandvars('$DATA_DIR2/small_bkgs/*')))
random.shuffle(neg_files)

#pos_dir = '$DATA_DIR/profile-car-100-40/*.png'
random.seed(0)
pos_dir = '$DATA_DIR/xi3zao3-car/*.png'
pos_files = sorted(glob.glob(os.path.expandvars(pos_dir)))
random.shuffle(pos_files)
pos_gen = itertools.cycle(pos_files)

def _load_cad_image(fn):
    im = gv.img.load_image(fn)
    im = gv.img.resize(im, (100, 100))
    gray_im, alpha = gv.img.asgray(im), im[...,3] 

    while alpha[0].sum() == 0:
        gray_im = gray_im[1:]
        alpha = alpha[1:] 

    while alpha[-1].sum() == 0:
        gray_im = gray_im[:-1]
        alpha = alpha[:-1]

    while alpha[:,0].sum() == 0:
        gray_im = gray_im[:,1:]
        alpha = alpha[:,1:]

    while alpha[:,-1].sum() == 0:
        gray_im = gray_im[:,:-1]
        alpha = alpha[:,:-1]

    return gray_im, alpha

def generate_nonoverlapping_positions(img_size, cad_size, num):
    bbs = []
    for _ in xrange(400):
        pos = tuple(random.randint(0, img_size[i]-cad_size[i]) for i in xrange(2))
        bb = (pos[0], pos[1], pos[0] + cad_size[0], pos[1] + cad_size[1])
        ok = True
        for other_bb in bbs:
            if gv.bb.area(gv.bb.intersection(bb, other_bb)) > 0:
                # No-go
                ok = False
                break 

        if ok:
            bbs.append(bb)
            yield bb
        
        if len(bbs) == num:
            break
         

path = os.path.expandvars('$CUSTOM_DIR')
img_path = os.path.join(path, str(name))
fout = open(os.path.join(path, '{0}.txt'.format(name)), 'w')

for img_id, fn in enumerate(neg_files[:limit]):
    bbs = []
    neg_im = gv.img.asgray(gv.img.load_image(fn))
    pos_im, alpha = _load_cad_image(pos_gen.next())

    for bb in generate_nonoverlapping_positions(neg_im.shape,  pos_im.shape, perimage):
        window = neg_im[bb[0]:bb[2],bb[1]:bb[3]]
        # This will update neg_im
        window[:] = window * (1 - alpha) + pos_im * alpha

        bbs.append(bb)

    # Save image
    gv.img.save_image(neg_im, os.path.join(img_path, 'test-{img_id}.png'.format(img_id=img_id)))

    boxes_str = " ".join(map(str, bbs))
    print("{img_id}: {boxes}".format(img_id=img_id, boxes=boxes_str), file=fout)
