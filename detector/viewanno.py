from __future__ import print_function
from __future__ import division

import matplotlib.pylab as plt
import gv
from plotting import plot_image

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='View image and its annotation')
    parser.add_argument('imgname', metavar='<image name>', nargs='?', type=int, help='Name of image in VOC repository')
    parser.add_argument('--class', dest='obj_class', default='bicycle', type=str, help='Object class for showing positives')
    parser.add_argument('--contest', type=str, choices=gv.datasets.datasets(), default='voc-val', help='Contest to try on')
    #parser.add_argument('-c', '--continue', action='store_true', help='List all')

    args = parser.parse_args()
    imagename = args.imgname
    obj_class = args.obj_class
    contest = args.contest
    
    if imagename is None:
        fileobjs, tot = gv.datasets.load_files(contest, obj_class)
        for f in fileobjs:
            if len(f.boxes) > 0:
                #print("{0:20} {1} ({2})".format(os.path.basename(f.path), len(f.boxes), sum([bbobj.difficult for bbobj in f.boxes])))
                print("Showing ", f.img_id)
                plot_image(f, filename='an/data-{0}.png'.format(f.img_id))
                #if raw_input("Continue? (Y/n): ") == 'n':
                #    break
                    
                
    else:
        fileobj = gv.datasets.load_file(contest, imagename, obj_class=obj_class)
        print(fileobj)
        plot_image(fileobj)
