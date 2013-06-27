from __future__ import absolute_import
import numpy as np
from .named_registry import NamedRegistry

class Saveable(object):
    @classmethod
    def load(cls, path):
        if path is None:
            return cls.load_from_dict({})
        else:
            d = np.load(path).flat[0]
            return cls.load_from_dict(d)
        
    def save(self, path):
        np.save(path, self.save_to_dict())

    @classmethod
    def load_from_dict(cls, d):
        raise NotImplementedError("Must override load_from_dict for Saveable interface")

    def save_to_dict(self):
        raise NotImplementedError("Must override save_to_dict for Saveable interface")


class SaveableRegistry(Saveable, NamedRegistry):
    @classmethod
    def load(cls, path):
        if path is None:
            return cls.load_from_dict({})
        else:
            d = np.load(path).flat[0]
            # Check class type
            class_name = d.get('name')
            if class_name is not None:
                return cls.getclass(class_name).load_from_dict(d)
            else:
                return cls.load_from_dict(d)

    def save(self, path):
        d = self.save_to_dict()
        d['name'] = self.name 
        np.save(path, d)
     
