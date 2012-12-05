
from patch_dictionary import PatchDictionary
import amitgroup as ag
import numpy as np

class Detector(object):
    """
    Detector
    """
    def __init__(self, num_mixtures, partsdict):
        assert isinstance(partsdict, PartsDictionary)
        self.partsdict = partsdict
        self.settings = {}
        self.num_mixtures = num_mixtures
        self.settings['bedges'] = parts.bedges_settings()
        self.modelclass = None
    
    def train_from_images(self, images):
        shape = None
        output = None
        #all_slices = []
        for i, filename in enumerate(images):
            edges = ag.features.bedges_from_images(images, **settings['bedges'])
            parts_list = self.partsdict.extract_parts(edges)
            if shape is None:
                shape = parts_list.shape
                output = np.empty((len(images),) + parts_list.shape)
                
            assert parts_list.shape == shape, "Images must all be of the same size"
            output[i] = parts_list
            #all_slices.append(parts_list)
            #size = tuple([max(size[i], parts_list.shape[i]) for i in xrange(2)])

        # Train mixture model OR SVM
        mixture = ag.stats.BernoulliMixture(self.num_mixtures, output)
        mixture.run_EM(1e-6)
        
        self.templates = mixture.templates 
    
        #self.modelclass = output 
        #return output
    
    @classmethod
    def load(self, path, partsdict):
        

    def save(self, path):
        np.savez(path, settings=settings
