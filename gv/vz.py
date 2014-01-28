import sys
import os

assert 'pylab' not in sys.modules, "Please import vz before importing pylab"
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
import gv
import itertools as itr

def imshow_raw(im):
    fn = os.path.expandvars('$PLOT_DIR/plot.png')

    gv.img.save_image(fn, im)

    os.chmod(fn, 0644)

def imshow(im):
    gv.imshow(im)

    fn = os.path.expandvars('$PLOT_DIR/plot.png')
    plt.savefig(fn)
    os.chmod(fn, 0644)

def image_grid(data, cmap=None, scale=1):
    if data.ndim == 3:
        data = data[:,np.newaxis]
    elif data.ndim != 4:
        assert "Image grid must have 3 or 4 dimensions"
    w, h = data.shape[:2]
    shape = data.shape[-2:]
    grid = gv.plot.ImageGrid(w, h, shape)

    for i, j in itr.product(xrange(w), xrange(h)):
        grid.set_image(data[i,j], i, j)

    fn = os.path.expandvars('$PLOT_DIR/plot.png')
    grid.save(fn, scale=scale)
    os.chmod(fn, 0644)
