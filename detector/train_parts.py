

from settings import argparse_settings 
sett = argparse_settings("Train parts") 

psettings = sett[sett['detector']['descriptor']]

import parser
import gv
import amitgroup as ag
import os.path
import random
import glob

ag.set_verbose(True)
if gv.parallel.main(__name__):

    path = os.path.expandvars(psettings['image_dir'])

    files = glob.glob(path)
    random.seed(0)
    random.shuffle(files)

    settings = dict(samples_per_image=psettings['samples_per_image'])
    cls = gv.BinaryDescriptor.getclass(sett['detector']['descriptor'])
    codebook = cls(psettings['part_size'], psettings['num_parts'], settings=psettings) 
    codebook.train_from_images(files[:psettings['num_images']])
    codebook.save(psettings['file'])

