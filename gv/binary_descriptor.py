from __future__ import absolute_import
from .saveable import SaveableRegistry

@SaveableRegistry.root
class BinaryDescriptor(SaveableRegistry):
    """
    This class is the base class of a binary descriptor. It should be able to
    take an image and spit out binary vectors of shape ``(X, Y, F)``, where ``(X, Y)``
    is the size of the image, and ``F`` the number of binary features.
    """
    def __init__(self, settings={}):
        self.settings = settings

    def extract_features(self, img):
        raise NotImplementedError("This is a base class and this function must be overloaded.")

    @property
    def num_features(self):
        raise NotImplementedError("This is a base class and this function must be overloaded.") 

    @property
    def subsample_size(self):
        raise NotImplementedError("This is a base class and this function must be overloaded.") 

    # Notice, user must also implement the Saveable interface! 
