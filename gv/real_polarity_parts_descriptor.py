from __future__ import division
import numpy as np
from ndfeature import ndfeature
from real_descriptor import RealDescriptor
from polarity_parts_descriptor import PolarityPartsDescriptor
from unraveled_hog import unraveled_hog

@RealDescriptor.register('polarity-parts')
class RealPolarityPartsDescriptor(RealDescriptor):
    def __init__(self, patch_size, num_parts, settings={}):
    #def __init__(self, settings={}):
        self._descriptor = PolarityPartsDescriptor(patch_size, num_parts, settings=settings)
        #self.settings.update(settings)

    @property
    def num_features(self):
        return self._descriptor.num_features

    @property
    def num_parts(self):
        return self._descriptor.num_parts

    @property
    def subsample_size(self):
        return self.settings['subsample_size']

    @property
    def settings(self):
        return self._descriptor.settings

    def extract_features(self, image, settings={}):
        feats = self._descriptor.extract_features(image, settings=settings)

        #new_feats = feats.astype(np.float64)
        #new_feats.upper = feats.upper
        #new_feats.lower = feats.lower
        #return new_feats
        return feats

    @classmethod
    def load_from_dict(cls, d):
        obj = RealPolarityPartsDescriptor(None, None)
        obj._descriptor = PolarityPartsDescriptor.load_from_dict(d)
        return obj

    def save_to_dict(self):
        return self._descriptor.save_to_dict()
