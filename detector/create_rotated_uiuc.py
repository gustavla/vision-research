from __future__ import division, print_function, absolute_import

import gv
import os
import numpy as np
from skimage import transform

if gv.parallel.main(__name__):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=str, help='Directory where to store the results')
    parser.add_argument('--angle', type=float, default=10.0, help='The most extreme angle')
    parser.add_argument('--duplicates', type=int, default=1)

    args = parser.parse_args()

    rad = args.angle * np.pi / 180.0
    duplicates = args.duplicates

    root_path = args.output
    img_path = os.path.join(root_path, 'TestImages')
    labels_path = os.path.join(root_path, 'trueLocations.txt')

    os.mkdir(root_path)
    os.mkdir(img_path)

    files, tot = gv.uiuc.load_testing_files()

    radians = np.linspace(-rad, rad, 20)

    max_img_id = np.max([fileobj.img_id for fileobj in files])

    all_bbs = [[] for _ in xrange(max_img_id+1)] 

    #matrices = [_translation_matrix(size/2, size/2) * _rotation_matrix(a) * _translation_matrix(-size/2, -size/2) for a in radians]

    rs = np.random.RandomState(0)

    for fileobj in files:
        for dup in xrange(duplicates):
            im = gv.img.load_image(fileobj.path)
            size = im.shape
            #print(fileobj)

            a = rs.uniform(-rad, rad)
            A = gv.matrix.translation(size[0]/2, size[1]/2) * gv.matrix.rotation(a) * gv.matrix.translation(-size[0]/2, -size[1]/2)

            # Rotate the image 
            deg = a * 180.0 / np.pi
            rot_im = transform.rotate(im, deg, resize=True)

            # How much do we need to adjust the position because of padding
            pad = ((rot_im.shape[0] - im.shape[0]) / 2, (rot_im.shape[1] - im.shape[1]) / 2)

            fn = os.path.basename(fileobj.path)
            name, ext = os.path.splitext(fn)
            new_path = os.path.join(img_path, '{}.png'.format(name))

            gv.img.save_image(new_path, rot_im)

            new_centers = []

            # Save rotated file

            # Rotate the centers of the bounding boxes
            for bbobj in fileobj.boxes:
                center = gv.bb.center(bbobj.box)
                x = np.matrix([center[0], center[1], 1]).T
                rot_x = A * x
                rot_x[:2] += np.asmatrix(pad).T
                #irot_x = [int(round(rot_x[i])) for i in xrange(2)]

                #half_size = (gv.bb.size(bbobj.box)[0]//2, gv.bb.size(bbobj.box)[1]//2)


                #bb = (irot_x[0] - half_size[0], irot_x[1]

                bb = gv.bb.create(center=np.asarray(rot_x).ravel()[:2], size=gv.bb.rotate_size(gv.bb.size(bbobj.box), deg))
                bb = [int(round(bbi)) for bbi in bb]

                #corner = (irot_x[0] - half_size[0], irot_x[1] - half_size[1])
                #print(bb)
                all_bbs[fileobj.img_id].append(bb)
            
    with open(labels_path, 'w') as f:
        for img_id, bbs in enumerate(all_bbs):
            print(bbs)
            s = ' '.join('({},{},{},{})'.format(*bb) for bb in bbs)
            print('{img_id}: {centers}'.format(img_id=img_id, centers=s), file=f)
