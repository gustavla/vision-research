from __future__ import division
import numpy as np
from ndfeature import ndfeature
from real_descriptor import RealDescriptor
from unraveled_hog import unraveled_hog

@RealDescriptor.register('hog')
class HOGDescriptor(RealDescriptor):
    def __init__(self, settings={}):
        self.settings = {}
        self.settings['cells_per_block'] = (3, 3)
        self.settings['pixels_per_cell'] = (6, 6)
        self.settings['orientations'] = 9 
        self.settings['polarity_sensitive'] = True
        self.settings['normalise'] = True 
        
        self.settings.update(settings)

    def extract_features(self, image, settings={}, raveled=True):
        from skimage import feature
        orientations = self.settings['orientations']
        ppc = self.settings['pixels_per_cell']
        if 0:
            hog = unraveled_hog(image, 
                              orientations=self.settings['orientations'],
                              pixels_per_cell=ppc,
                              cells_per_block=self.settings['cells_per_block'],
                              normalise=self.settings['normalise'])

        from gv.hog import hog as hogf
        X = np.tile(image[...,np.newaxis], 3)
        image = np.asarray(X, dtype=np.double)
        hog = hogf(image, sbin=4) 

        if not self.settings['polarity_sensitive']:
            assert self.settings['orientations'] % 2 == 0, "Must have even number of orientations for polarity insensitive edges"
            S = self.settings['orientations'] // 2
            hog = (hog[...,:S] + hog[...,S:]) / 2

        # Let's binarize the features somehow
        #hog = (hog > self.settings['binarize_threshold']).astype(np.uint8)

        if raveled:
            hog = hog.reshape(hog.shape[:2] + (-1,))

        # How much space was cut away?
        buf = tuple(image.shape[i] - hog.shape[i] * ppc[i] for i in xrange(2))
        lower = tuple(buf[i]//2 for i in xrange(2))
        upper = tuple(image.shape[i] - (buf[i]-lower[i]) for i in xrange(2))

        return ndfeature(hog, lower=lower, upper=upper)

    @property
    def num_features(self):
        orients = self.settings['orientations']
        if not self.settings['polarity_sensitive']:
            orients //= 2
        return orients * np.prod(self.settings['cells_per_block'])

    @property
    def subsample_size(self):
        return self.settings['pixels_per_cell']

    def save_to_dict(self):
        return self.settings

    @classmethod
    def load_from_dict(cls, d):
        return cls(d)
