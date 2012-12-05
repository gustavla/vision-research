
import argparse

parser = argparse.ArgumentParser(description='Pose images to a single size')
parser.add_argument('side', type=int, help="All images will be arranged to this size")

args = parser.parse_args()
side = args.side

x = [1,2,3]

import ipdb; ipdb.set_trace()


from config import SETTINGS
import glob
import os.path
from superimpose import find_bounding_box

for f in glob.glob(os.path.join(SETTINGS['src_dir'], '*.png')):
    im = np.load(f) 
    bb = find_bounding_box(im)
    
    # Resize to side
    

