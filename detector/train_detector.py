
import argparse

parser = argparse.ArgumentParser(description='Train mixture model on edge data')
parser.add_argument('patches', metavar='<patches file>', type=argparse.FileType('rb'), help='Filename of patches file')
parser.add_argument('model', metavar='<output model file>', type=argparse.FileType('wb'), help='Filename of the output models file')
parser.add_argument('mixtures', metavar='<number mixtures>', type=int, help='Number of mixture components')

args = parser.parse_args()
patches_file = args.patches
model_file = args.model
num_mixtures = args.mixtures

import gv
from config import SETTINGS
import glob
import os.path
import amitgroup as ag

ag.set_verbose(True)

patch_dict = gv.PatchDictionary.load(patches_file)
detector = gv.Detector(num_mixtures, patch_dict)

files = glob.glob(os.path.join(SETTINGS['src_dir'], "*.png"))

#files = files[:10]

detector.train_from_images(files)

detector.save(model_file)
