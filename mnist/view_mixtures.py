
import amitgroup as ag
import numpy as np
import sys
from itertools import product

try:
    filename = sys.argv[1]
    letter = int(sys.argv[2])
except IndexError: 
    print "<mixture file> <letter> [<rotation>]"
    sys.exit(0)

try:
    rotation = int(sys.argv[3])
except IndexError: 
    rotation = None
    
assert 0 <= letter <= 9

data = np.load(filename)
#assert data['templates'] and data['weights']
templates = data['templates']
meta = data['meta'].flat[0]

if rotation is None:
    M = meta['mixtures'] 
    J = templates.shape[2]
    ag.plot.images([
        templates[letter,m,j] for m, j in product(range(M), range(J))
    ], subplots=(M, J))
else:
    assert 0 <= rotation <= 8
    ag.plot.images(templates[letter,:,:,:,rotation])


