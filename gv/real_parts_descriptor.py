from __future__ import division, print_function, absolute_import
import numpy as np
from .ndfeature import ndfeature
from .real_descriptor import RealDescriptor
from .parts_descriptor import PartsDescriptor
from .unraveled_hog import unraveled_hog

@RealDescriptor.register('parts')
class RealPartsDescriptor(RealDescriptor):
    def __init__(self, patch_size, num_parts, settings={}):
        self._descriptor = PartsDescriptor(patch_size, num_parts, settings=settings)

    @property
    def num_features(self):
        return self._descriptor.num_features

    @property
    def num_parts(self):
        return self._descriptor.num_parts

    @property
    def num_true_parts(self):
        return self._descriptor.num_parts

    @property
    def subsample_size(self):
        return self.settings['subsample_size']

    @property
    def settings(self):
        return self._descriptor.settings

    def extract_features(self, image, settings={}, *args, **kwargs):
        feats = self._descriptor.extract_features(image, settings=settings, *args, **kwargs)

        #new_feats = feats.astype(np.float64)
        #new_feats.upper = feats.upper
        #new_feats.lower = feats.lower
        #return new_feats
        return feats

    @classmethod
    def load_from_dict(cls, d):
        obj = RealPartsDescriptor(None, None)
        obj._descriptor = PartsDescriptor.load_from_dict(d)
        return obj

    def save_to_dict(self):
        return self._descriptor.save_to_dict()

    @property
    def visparts(self):
        return self._descriptor.visparts
