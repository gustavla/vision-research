from __future__ import print_function
from __future__ import division

import matplotlib.pylab as plt
import gv
from config import VOCSETTINGS
from plotting import plot_results

def show_image(fileobj):
    img = gv.img.load_image(fileobj.path)

    plt.imshow(img)

    for bbobj in fileobj.boxes:
        bb = bbobj.box
        if bbobj.difficult:
            color = 'red'
        else:
            color = 'lightgreen'
        plt.gca().add_patch(plt.Rectangle((bb[1], bb[0]), bb[3]-bb[1], bb[2]-bb[0], facecolor='none', edgecolor=color, linewidth=2.0))
        #plt.gca().add_patch(plt.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1], facecolor='none', edgecolor='lightgreen', linewidth=2.0))

    plt.show()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='View image and its annotation')
    parser.add_argument('imgname', metavar='<image name>', nargs='?', type=int, help='Name of image in VOC repository')
    #parser.add_argument('-c', '--continue', action='store_true', help='List all')

    args = parser.parse_args()
    imagename = args.imgname
    
    if imagename is None:
        fileobjs, tot = gv.voc.load_training_files(VOCSETTINGS, 'bicycle')
        for f in fileobjs:
            if len(f.boxes) > 0:
                #print("{0:20} {1} ({2})".format(os.path.basename(f.path), len(f.boxes), sum([bbobj.difficult for bbobj in f.boxes])))
                print("Showing ", f.img_id)
                show_image(f)
                #if raw_input("Continue? (Y/n): ") == 'n':
                #    break
                    
                
    else:
        fileobj = gv.voc.load_training_file(VOCSETTINGS, 'bicycle', imagename)
        show_image(fileobj)
