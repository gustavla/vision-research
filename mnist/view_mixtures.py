
import amitgroup as ag
import numpy as np
import sys
from itertools import product

try:
    filename = sys.argv[1]
    letter = int(sys.argv[2])
except IndexError: 
    print "<mixture file> <letter>"
    sys.exit(0)

try:
    rotation = int(sys.argv[3])
except IndexError: 
    rotation = None
    
assert 0 <= letter <= 9

data = np.load(filename)
#assert data['templates'] and data['weights']
templates = data['templates']
graylevel_templates = dict(data).get('graylevel_templates')
meta = data['meta'].flat[0]

if rotation is None:
    M = meta['mixtures'] 
    # Display gray level templates if they are stored
    if graylevel_templates is not None:
        ag.plot.images(graylevel_templates[letter])
    else:
        J = templates.shape[2]
        ag.plot.images([
            templates[letter,m,j] for m, j in product(range(M), range(J))
        ], subplots=(M, J))
else:
    raise NotImplementedError("Broken")
    assert 0 <= rotation < 8
    ag.plot.images(templates[letter,:,:,:,rotation])


