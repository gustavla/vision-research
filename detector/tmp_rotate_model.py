from __future__ import division, print_function, absolute_import
#from pnet.vzlog import default as vz
import pylab as plt
import gv
import numpy as np
from skimage.transform import rotate
from copy import copy
from gv.keypoints import get_key_points

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')
    parser.add_argument('output_model', metavar='<output model file>', type=str, help='Filename of output model file')

    args = parser.parse_args()

    d = gv.BernoulliDetector.load(args.model)

    assert len(d.kernel_templates) == 1, "Can only rotate a model that has a single component to begin with"

    deg_per_step = d.descriptor.degrees_per_step
    ROT = d.descriptor.settings.get('orientations', 1)
    print('degrees per step', deg_per_step)

    rots = [-ROT//4, ROT//4]
    new_components = []

    #kern = d.kernel_templates[0]

    w0 = d.weights(0)

    weights = [w0]

    bbs = copy(d.extra['bbs'])
    bb0 = bbs[0]
    supports = copy(d.support)
    kernel_sizes = copy(d.kernel_sizes)

    for rot in rots:
        deg = rot * deg_per_step
        print('deg', deg)
        slices = []
        for f in xrange(w0.shape[-1]):
            rotted = (rotate(w0[...,f] / 10 + 0.5, deg, resize=True, cval=0.5) - 0.5) * 10

            slices.append(rotted)

            if f % 50 == 0:
                print(f)

            # Crop it a bit
            # TODO
            
            if 0:    
                plt.figure()
                plt.imshow(w0[...,f], vmin=-3, vmax=3, cmap=plt.cm.RdBu_r, interpolation='nearest')
                plt.savefig(vz.generate_filename())

                plt.figure()
                plt.imshow(rotted, vmin=-3, vmax=3, cmap=plt.cm.RdBu_r, interpolation='nearest')
                plt.savefig(vz.generate_filename())


        # THE PARTS ARE NOT ROTATED YET!
        for k in xrange(w0.shape[-1]//ROT):
            slices[k*ROT:(k+1)*ROT] = np.roll(slices[k*ROT:(k+1)*ROT], rot)

        slices = np.rollaxis(np.asarray(slices), 0, 3)

        weights.append(slices)

        

        bb = gv.bb.create(center=gv.bb.center(bb0), size=gv.bb.rotate_size(gv.bb.size(bb0), deg))
        bbs.append(bb)
        supports.append(d.support[0])
        kernel_sizes.append(d.kernel_sizes[0])

    d.num_mixtures = len(weights)

    print(map(np.shape, weights))

    # Invent new keypoints    
    indices = []
    for m in xrange(d.num_mixtures):
        w = weights[m]
        ii = get_key_points(w, suppress_radius=d.settings.get('indices_suppress_radius', 4), even=True)
        indices.append(ii)

    # Now store the weights preprocessed
    d.indices = indices
    d.extra['weights'] = weights
    d.extra['bbs'] = bbs
    print('bbs', bbs)
    d.support = supports
    d.kernel_sizes = kernel_sizes
    print('d.TEMP_second', d.TEMP_second)
    print('kernel_sizes', d.kernel_sizes)
    d.TEMP_second = False
    d.save(args.output_model)

    #vz.finalize()


if __name__ == '__main__':
    main()
