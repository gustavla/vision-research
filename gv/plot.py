from __future__ import division
import numpy as np
#from scipy.ndimage.interpolation import zoom

class PlotImage(object):
    def __init__(self, rows, cols, size, border_color=np.array([0.5, 0.5, 0.5])):
        self._rows = rows
        self._cols = cols
        self._size = size
        self._border = 1
        self._border_color

        self._fullsize = (self._border + (size[0] + self._border) * self._rows,
                          self._border + (size[1] + self._border) * self._cols)

        self._data = np.ones(self._fullsize + (3,), dtype=np.float64)

    def set_image(self, row, col, image, vmin=None, vmax=None, cmap=None):
        import matplotlib as plt
        if cmap is None:
            cmap = plt.cm.gray
        if vmin is None:
            vmin = image.min()
        if vmax is None:
            vmax = image.max()

        from gv.fast import resample_and_arrange_image

        image_indices = (np.clip((image - vmin) / vmax, 0, 1) * 255).astype(np.uint8)

        assert hasattr(cmap, 'lut_'), "Not a proper cmap given"
        rgb = resample_and_arrange_image(image_indices, self._size, cmap)

        self._data[row * (self._size[0] + self._border) : (row + 1) * (self._size[0] + self._border) + self._border,
                   col * (self._size[1] + self._border) : (col + 1) * (self._size[1] + self._border) + self._border] = self._border_color 

        anchor = (self._border + row * (self._size[0] + self._border),
                  self._border + col * (self._size[1] + self._border))

        self._data[anchor[0] : anchor[0] + rgb.shape[0],
                   anchor[1] : anchor[1] + rgb.shape[1]] = rgb

    def save(self, path):
        gv.img.save_image(path, self._data)
