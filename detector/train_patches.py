
import argparse

parser = argparse.ArgumentParser(description='Train mixture model on edge data')
parser.add_argument('output', metavar='<patches file>', type=argparse.FileType('wb'), help='Filename of patches file')

args = parser.parse_args()
output_file = args.output

import parser
import gv
import amitgroup as ag
import os.path
import random
import glob

from config import SETTINGS

ag.set_verbose(True)

files = glob.glob(os.path.join(SETTINGS['patches_dir'], "*.jpg"))
print files[:10]
random.seed(0)
random.shuffle(files)

settings = dict(samples_per_image=100)
codebook = gv.PatchDictionary((5, 5), 200) 
codebook.train_from_images(files[:500])

codebook.save(output_file)
