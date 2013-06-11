from __future__ import absolute_import
from .binary_descriptor import BinaryDescriptor
import amitgroup as ag
import amitgroup.features

@BinaryDescriptor.register('edges')
class EdgeDescriptor(BinaryDescriptor):
    def __init__(self, settings={}):
        self.settings = {}
        self.settings['contrast_insensitive'] = False
        self.settings['k'] = 5 
        self.settings['radius'] = 1
        self.settings['minimum_contrast'] = 0.1
        self.settings.update(settings)

    def extract_features(self, img, settings={}):
        #return ag.features.bedges_from_image(img, **self.settings)
        sett = self.settings.copy()
        sett.update(settings)
        if 'spread_radii' in sett:
            del sett['spread_radii']
        if 'crop_border' in sett:
            del sett['crop_border']
        return ag.features.bedges(img, **sett)

    @property
    def num_features(self):
        return 4 if self.settings['contrast_insensitive'] else 8

    def save_to_dict(self):
        return self.settings

    @classmethod
    def load_from_dict(cls, d):
        return cls(d)
