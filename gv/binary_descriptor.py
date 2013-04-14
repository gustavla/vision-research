from __future__ import absolute_import
from .saveable import Saveable

_DESCRIPTORS = {}

class BinaryDescriptor(Saveable):
    """
    This class is the base class of a binary descriptor. It should be able to
    take an image and spit out binary vectors of shape ``(X, Y, F)``, where ``(X, Y)``
    is the size of the image, and ``F`` the number of binary features.

    They can also optionally specify a pooling routine
    """
    def __init__(self, settings={}):
        self.settings = settings

    def extract_features(self, img):
        raise NotImplementedError("This is a base class and this function must be overloaded.")

    @property
    def num_features(self):
        raise NotImplementedError("This is a base calss and this function must be overloaded.") 

    @property
    def name(self):
        # Automatically overloaded by 'register'
        return "noname" 

    # Leaving this as they are means no pooling
    #def pooling_size(self):
    #    return 1

    #def pool_features(self, feats):
    #    return feats

    # Notice, must also implement the Saveable interface! 

    # These functions are for managing the register of binary descriptors
    @classmethod
    def register(cls, name):
        def register_decorator(reg_cls):
            def name_func(self):
                return name
            reg_cls.name = property(name_func)
            assert issubclass(reg_cls, cls), "Must be subclass of BinaryDescriptor"
            global _DESCIPTORS
            _DESCRIPTORS[name] = reg_cls
        return register_decorator

    @classmethod
    def getclass(cls, name):
        global _DESCRIPTORS
        return _DESCRIPTORS[name]

    @classmethod
    def construct(cls, name, *args, **kwargs):
        global _DESCRIPTORS
        return _DESCRIPTORS[name](*args, **kwargs)

    @classmethod
    def descriptors(cls):
        global _DESCRIPTORS
        return _DESCRIPTORS
