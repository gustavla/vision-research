import glob
import os.path
import numpy as np
from PIL import Image
from config import SETTINGS
from superimpose import find_bounding_box

# This file is barely started, but will eventually tighten training data.
# It should not tighten too much though, as things can get cropped away in the
# pooling steps

infiles = glob.glob(os.path.join(SETTINGS['src_dir'], '*.jpg'))

bounding_box = [np.inf, np.inf, -np.inf, -np.inf]
for path in infiles:
    im = Image.open(path)
    bb = find_bounding_box(im)

    bounding_box[0] = min(bb[0], bounding_box[0])
    bounding_box[1] = min(bb[1], bounding_box[1])
    bounding_box[2] = max(bb[2], bounding_box[2])
    bounding_box[3] = max(bb[3], bounding_box[3])
        
    #fn = os.path.basename(path) 

print bounding_box
    
    
