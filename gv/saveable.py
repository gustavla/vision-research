import numpy as np

class Saveable(object):
    @classmethod
    def load(cls, path):
        d = np.load(path).flat[0]
        return cls.load_from_dict(d)
        
    def save(self, path):
        np.save(path, self.save_to_dict())

    @classmethod
    def load_from_dict(cls, d):
        raise NotImplementedError("Must override load_from_dict for Saveable interface")

    def save_to_dict(self):
        raise NotImplementedError("Must override save_to_dict for Saveable interface")

