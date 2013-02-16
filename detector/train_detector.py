

from settings import argparse_settings
sett = argparse_settings("Train detector")
dsettings = sett['detector']

#import argparse

#parser = argparse.ArgumentParser(description='Train mixture model on edge data')
#parser.add_argument('patches', metavar='<patches file>', type=argparse.FileType('rb'), help='Filename of patches file')
#parser.add_argument('model', metavar='<output model file>', type=argparse.FileType('wb'), help='Filename of the output models file')
#parser.add_argument('mixtures', metavar='<number mixtures>', type=int, help='Number of mixture components')
#parser.add_argument('--use-voc', action='store_true', help="Use VOC data to train model")

import gv
import glob
import os.path
import amitgroup as ag

ag.set_verbose(True)

#patch_dict = gv.PatchDictionary.load(patches_file)
des_name = dsettings['descriptor']
descriptor_filename = sett[des_name].get('file')
descriptor_cls = gv.BinaryDescriptor.getclass(des_name)
if descriptor_filename is None:
    # If there is no descriptor filename, we'll just build it from the settings
    descriptor = descriptor_cls.load_from_dict(sett[des_name])
else:
    descriptor = descriptor_cls.load(descriptor_filename)

detector = gv.Detector(dsettings['num_mixtures'], descriptor, dsettings)

if dsettings['use_voc']:
    files = gv.voc.load_object_images_of_size(sett['voc'], 'bicycle', dsettings['image_size'], dataset='train')
else:
    files = glob.glob(os.path.join(dsettings['cad_dir'], "*.png"))

limit = dsettings.get('limit_images')
if limit is not None:
    files = files[:limit]

print "Training on {0} files".format(len(files))
#files = files[:10]

detector.train_from_images(files)

detector.save(dsettings['file'])

