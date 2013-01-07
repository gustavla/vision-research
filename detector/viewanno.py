from __future__ import print_function
from __future__ import division

import matplotlib.pylab as plt
import gv
from config import VOCSETTINGS
from plotting import plot_results

def show_image(imagename):
    fileobj = gv.voc.load_training_file(VOCSETTINGS, 'bicycle', imagename)
    if fileobj is None:
        print("Could not find image", file=sys.stderr)
        sys.exit(0)

    img = gv.img.load_image(fileobj.path)

    plt.imshow(img)

    for bbobj in fileobj.boxes:
        bb = bbobj.box
        plt.gca().add_patch(plt.Rectangle((bb[1], bb[0]), bb[3]-bb[1], bb[2]-bb[0], facecolor='none', edgecolor='lightgreen', linewidth=2.0))
        #plt.gca().add_patch(plt.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], facecolor='none', edgecolor='lightgreen', linewidth=2.0))

    plt.show()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='View image and its annotation')
    parser.add_argument('imgname', metavar='<image name>', type=int, help='Name of image in VOC repository')
    parser.add_argument('-c', '--continue', action='store_true', help='List all')
    args = parser.parse_args()
    imagename = args.imgname

    show_image(imagename)
