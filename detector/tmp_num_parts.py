
import argparse

parser = argparse.ArgumentParser(description='Train mixture model on edge data')
parser.add_argument('parts', metavar='<parts file>', type=argparse.FileType('rb'), help='Filename of parts file')

args = parser.parse_args()
parts_file = args.parts

import numpy as np
import gv

parts_descriptor = gv.BinaryDescriptor.getclass('parts').load(parts_file)
print 'num_parts', parts_descriptor.num_features
