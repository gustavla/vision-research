
import amitgroup as ag
import numpy as np
import sys

try:
    filename = sys.argv[1]
    letter = int(sys.argv[2])
    rotation = int(sys.argv[3])
except IndexError: 
    print "<file> <letter> <rotation>"
    sys.exit(0)
    
assert 0 <= letter <= 9
assert 0 <= rotation <= 8

data = np.load(filename)
#assert data['templates'] and data['weights']
templates = data['templates']

ag.plot.images(templates[letter,:,:,:,rotation])


