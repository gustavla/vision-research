from __future__ import division
import numpy as np
import gv
#from scipy.ndimage.interpolation import zoom

class ImageGrid(object):
    def __init__(self, rows, cols, size, border_color=np.array([0.5, 0.5, 0.5])):
        self._rows = rows
        self._cols = cols
        self._size = size
        self._border = 1
        self._border_color = np.array(border_color)

        self._fullsize = (self._border + (size[0] + self._border) * self._rows,
                          self._border + (size[1] + self._border) * self._cols)

        self._data = np.ones(self._fullsize + (3,), dtype=np.float64)

    @property
    def image(self):
        return self._data

    def set_image(self, image, row, col, vmin=None, vmax=None, cmap=None):
        import matplotlib as mpl
        import matplotlib.pylab as plt

        if cmap is None:
            cmap = plt.cm.gray
        if vmin is None:
            vmin = image.min()
        if vmax is None:
            vmax = image.max()

        from gv.fast import resample_and_arrange_image

        if vmin == vmax:
            diff = 1
        else:
            diff = vmax - vmin
            
        image_indices = (np.clip((image - vmin) / diff, 0, 1) * 255).astype(np.uint8)

        rgb = resample_and_arrange_image(image_indices, self._size, mpl.colors.makeMappingArray(256, cmap))

        self._data[row * (self._size[0] + self._border) : (row + 1) * (self._size[0] + self._border) + self._border,
                   col * (self._size[1] + self._border) : (col + 1) * (self._size[1] + self._border) + self._border] = self._border_color 

        anchor = (self._border + row * (self._size[0] + self._border),
                  self._border + col * (self._size[1] + self._border))

        self._data[anchor[0] : anchor[0] + rgb.shape[0],
                   anchor[1] : anchor[1] + rgb.shape[1]] = rgb

    def save(self, path, scale=1):
        data = self._data
        if scale != 1:
            from skimage.transform import resize
            data = resize(self._data, tuple([self._data.shape[i] * scale for i in xrange(2)]), order=0)
        gv.img.save_image(path, data)
