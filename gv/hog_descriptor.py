
import numpy as np
from .ndfeature import ndfeature
from .real_descriptor import RealDescriptor
from .unraveled_hog import unraveled_hog

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

    def extract_features(self, image, settings={}, must_preserve_size=False, raveled=True):
        sett = self.settings.copy()
        sett.update(settings)

        from skimage import feature
        orientations = sett['orientations']
        ppc = self.settings['pixels_per_cell']
        if 0:
            hog = unraveled_hog(image, 
                              orientations=sett['orientations'],
                              pixels_per_cell=ppc,
                              cells_per_block=sett['cells_per_block'],
                              normalise=sett['normalise'])

        from gv.hog import hog as hogf
        X = np.tile(image[...,np.newaxis], 3)
        image = np.asarray(X, dtype=np.double)
        hog = hogf(image, sbin=self.subsample_size[0]) 

        #hog = hog[...,:9]

        if 0:
            new_hog = np.concatenate([
                hog[...,:9],
                (hog[...,-4:] * np.array([1,1,1,1])).mean(axis=-1)[...,np.newaxis],
                (hog[...,-4:] * np.array([-1,-1,1,1])).mean(axis=-1)[...,np.newaxis],
                (hog[...,-4:] * np.array([-1,1,-1,1])).mean(axis=-1)[...,np.newaxis],
                (hog[...,-4:] * np.array([-1,1,1,-1])).mean(axis=-1)[...,np.newaxis],
            ], axis=2)

            hog = new_hog

        if 0:
            # The HOG we're using is not that configurable. It uses polarity insensitive gradients.
            print(('orientations', orientations))
            if not sett['polarity_sensitive']:
                assert orientations % 2 == 0, "Must have even number of orientations for polarity insensitive edges"
                S = orientations // 2
                hog = (hog[...,:S] + hog[...,S:]) / 2

        # Let's binarize the features somehow
        #hog = (hog > self.settings['binarize_threshold']).astype(np.uint8)

        #if raveled:
            #hog = hog.reshape(hog.shape[:2] + (-1,))

        cb = sett.get('crop_border')
        if cb:
            # Due to spreading, the area of influence can be greater
            # than what we're cutting off. That's why it's good to have
            # a cut_border property if you're training on real images.
            hog = hog[cb:-cb, cb:-cb]

        # How much space was cut away?
        buf = tuple(image.shape[i] - hog.shape[i] * ppc[i] for i in range(2))
        lower = tuple(buf[i]//2 for i in range(2))
        upper = tuple(image.shape[i] - (buf[i]-lower[i]) for i in range(2))

        return ndfeature(hog, lower=lower, upper=upper)

    @property
    def num_features(self):
        #orients = self.settings['orientations']
        #if not self.settings['polarity_sensitive']:
            #orients //= 2
        #return orients * np.prod(self.settings['cells_per_block'])
        return 13

    @property
    def subsample_size(self):
        return self.settings['pixels_per_cell']

    def save_to_dict(self):
        return self.settings

    @classmethod
    def load_from_dict(cls, d):
        return cls(d)
