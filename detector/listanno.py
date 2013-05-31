from __future__ import print_function
import gv
from config import VOCSETTINGS
import os.path
import argparse

parser = argparse.ArgumentParser(description='List files and number of annotations of a certain object')
parser.add_argument('object', type=str, help='Object class to list')
parser.add_argument('-a', '--all', action='store_true', help='List all')

args = parser.parse_args()
object_class = args.object
listall = args.all

fileobjs, tot = gv.voc.load_training_files(VOCSETTINGS, object_class, dataset='test')

print("<filename> <number of boxes> (<number of which are difficult>)")
for f in fileobjs:
    if len(f.boxes) > 0 or listall:
        print("{0:20} {1} ({2})".format(os.path.basename(f.path), len(f.boxes), sum([bbobj.difficult for bbobj in f.boxes])))

print("Total number: {0}".format(tot))
