

from settings import argparse_settings 
sett = argparse_settings("Train parts") 
psettings = sett['parts']

import parser
import gv
import amitgroup as ag
import os.path
import random
import glob

from config import SETTINGS

ag.set_verbose(True)

files = glob.glob(os.path.join(psettings['image_dir'], "*.jpg"))
random.seed(0)
random.shuffle(files)

settings = dict(samples_per_image=psettings['samples_per_image'])
cls = gv.BinaryDescriptor.getclass('parts')
codebook = cls(psettings['part_size'], psettings['num_parts'], settings=psettings) 
codebook.train_from_images(files[:psettings['num_images']])
codebook.save(psettings['file'])
