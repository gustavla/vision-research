
import argparse

parser = argparse.ArgumentParser(description='Test response of model')
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')
parser.add_argument('img', metavar='<image file>', type=argparse.FileType('rb'), help='Filename of image file')

args = parser.parse_args()
model_file = args.model
image_file = args.img

import gv
import numpy as np
from PIL import Image

detector = gv.Detector.load(model_file)

img = np.array(Image.open(image_file)).astype(np.float64) / 255.0

x = detector.response_map(img)

np.save('x', x)
