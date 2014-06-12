import sys
import os

print("LOADING VZ")
assert 'pylab' not in sys.modules, "Please import vz before importing pylab"
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
import gv
import itertools as itr
import numpy as np
from matplotlib.pylab import cm

__all__  = [
        'ismhow_raw',
        'imshow',
        'image_grid',
        'cm']


def imshow_raw(im, name='plot'):
    fn = os.path.expandvars('$PLOT_DIR/{}.png'.format(name))

    gv.img.save_image(fn, im)

    os.chmod(fn, 0o644)

def imshow(im, name='plot'):
    gv.imshow(im)

    fn = os.path.expandvars('$PLOT_DIR/{}.png'.format(name))
    plt.savefig(fn)
    os.chmod(fn, 0o644)

def image_grid(data, cmap=None, vmin=None, vmax=None, scale=1, name='plot'):
    if data.ndim == 3:
        data = data[:,np.newaxis]
    elif data.ndim != 4:
        assert "Image grid must have 3 or 4 dimensions"
    w, h = data.shape[:2]
    shape = data.shape[-2:]
    grid = gv.plot.ImageGrid(w, h, shape)

    for i, j in itr.product(range(w), range(h)):
        grid.set_image(data[i,j], i, j, vmin=vmin, vmax=vmax, cmap=cmap)

    fn = os.path.expandvars('$PLOT_DIR/{}.png'.format(name))
    grid.save(fn, scale=scale)
    os.chmod(fn, 0o644)
