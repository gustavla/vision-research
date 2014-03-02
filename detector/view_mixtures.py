from __future__ import division

import gv
import os
import numpy as np
import math

def view_mixtures(detector, output_file=None):
    import amitgroup as ag
    import matplotlib.pylab as plt
    from matplotlib.patches import Rectangle
    data = None
    if detector.support is None:
        return
        # Visualize feature activity if the support does not exist
        #assert 0, "This is broken since refactoring"
        data = detector.kernel_templates.sum(axis=-1)# / detector.kernel_templates.shape[-1] 
        data /= data.max()
        zero_to_one = False
    else:
        data = detector.support
        zero_to_one = True

    try:
        w = detector.mixture.weights[i]
    except:
        w = -1

    fig = plt.figure(figsize=(12, 6)) 
    for m, datapoint in enumerate(data):
        ax = fig.add_subplot(2, math.ceil(len(data)/2), m+1)
        ax.set_axis_off()
        ax.imshow(datapoint, vmin=0, vmax=1, interpolation='nearest', cmap=plt.cm.gray)
        ax.set_title(str(m))

        bb = np.multiply(detector.boundingboxes[m], np.tile(detector.settings['subsample_size'], 2))
        ax.add_patch(Rectangle((bb[1], bb[0]), bb[3]-bb[1], bb[2]-bb[0], facecolor='none', edgecolor='cyan', alpha=0.4, linewidth=2.0))

    if output_file is not None:
        plt.savefig(output_file)
        plt.close()
    else:
        plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='View mixture components')
    parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')
    parser.add_argument('-o', '--output', type=argparse.FileType('wb'), help='Filename of output image')

    import matplotlib as mpl

    args = parser.parse_args()
    model_file = args.model

    if args.output is not None:
        mpl.use('Agg')

    # Load detector
    detector = gv.Detector.load(model_file)

    view_mixtures(detector, output_file=args.output)
    os.chmod(args.output.name, 0644)
