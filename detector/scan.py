
import argparse

parser = argparse.ArgumentParser(description='Test response of model')
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')
parser.add_argument('img', metavar='<image file>', type=argparse.FileType('rb'), help='Filename of image file')

args = parser.parse_args()
model_file = args.model
image_file = args.img

import gv

detector = gv.Detector.load(model_file)

x = detector.response_map(img)

print x.shape
