from __future__ import division

import gv
import os

def view_mixtures(detector, output_file=None):
    import amitgroup as ag
    import matplotlib.pylab as plt
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

    #print zero_to_one

    #import pdb; pdb.set_trace()

    #ag.plot.images(data, zero_to_one=True, caption=lambda i, im: "{0}: max: {1:.02} (w: {2:.02})".format(i, im.max(), detector.mixture.weights[i]), show=False)
    ag.plot.images(data, zero_to_one=True, caption=lambda i, im: "{0}: max: {1:.02} (w: {2:.02})".format(i, im.max(), 1.0))
    #ag.plot.images(data, zero_to_one=True, caption=lambda i, im: "")
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
