
import argparse

parser = argparse.ArgumentParser(description='Train mixture model on edge data')
parser.add_argument('patches', metavar='<patches file>', type=argparse.FileType('b'), help='Filename of patches file')
parser.add_argument('model', metavar='<output model file>', type=argparse.FileType('wb'), help='Filename of the output models file')

args = parser.parse_args()
patches_file = args.output
model_file = args.model

import gv

patch_dict = gv.PatchDictionary.load(patches_file)

