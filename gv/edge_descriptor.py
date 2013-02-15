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

    def extract_features(self, img):
        #return ag.features.bedges_from_image(img, **self.settings)
        return ag.features.bedges(img, **self.settings)

    def save_to_dict(self):
        return self.settings

    @classmethod
    def load_from_dict(cls, d):
        return cls(d)
