
from patch_dictionary import PatchDictionary
import amitgroup as ag
import numpy as np
import scipy.signal
from saveable import Saveable

def max_pooling(data, size):
    steps = tuple([data.shape[i]//size[i] for i in xrange(2)])
    output = np.zeros(steps + (data.shape[-1],))
    for i in xrange(steps[0]):
        for j in xrange(steps[1]):
            output[i,j] = data[i*size[0]:(i+1)*size[0], j*size[1]:(j+1)*size[1]].max(axis=0).max(axis=0)
    return output

class Detector(Saveable):
    """
    Detector
    """
    def __init__(self, num_mixtures, patch_dict):
        assert isinstance(patch_dict, PatchDictionary)
        self.patch_dict = patch_dict
        self.settings = {}
        self.num_mixtures = num_mixtures
        self.mixture = None
    
    def train_from_images(self, images):
        shape = None
        output = None
        #all_slices = []
        for i, filename in enumerate(images):
            ag.info(i, "Processing file", filename)
            edges = ag.features.bedges_from_image(filename, **self.patch_dict.bedges_settings())
            print 'edges.shape', edges.shape
            parts_list = self.patch_dict.extract_parts(edges)
        
            # Do some max pooling here.
            small = max_pooling(parts_list, (8, 8))

            print 'small.shape', small.shape

            if shape is None:
                shape = small.shape
                output = np.empty((len(images),) + small.shape)
                
            assert small.shape == shape, "Images must all be of the same size"
            output[i] = small 
            #all_slices.append(parts_list)
            #size = tuple([max(size[i], parts_list.shape[i]) for i in xrange(2)])

        assert output is not None

        print 'output.shape', output.shape
    
        ag.info("Running mixture model in Detector")

        # Train mixture model OR SVM
        mixture = ag.stats.BernoulliMixture(self.num_mixtures, output)
        mixture.run_EM(1e-6)
        
        #self.templates = mixture.templates
        self.mixture = mixture
        # TODO: How to store the mixture model the best way?

    def response_map(self, image):
        """Retrieves log-likelihood response on 'image' (no scaling done)"""
        edges = ag.features.bedges_from_image(image, **self.patch_dict.bedges_settings())
        parts = self.patch_dict.extract_parts(edges)
        small = max_pooling(parts, (8, 8)) 
        
        res = None
        for k in xrange(self.num_mixtures):
            for f in xrange(small.shape[-1]):
                print small.shape, self.mixture.log_templates.shape
                r1 = scipy.signal.convolve2d(small[...,f], self.mixture.log_templates[k,...,f])
                r2 = scipy.signal.convolve2d(1-small[...,f], self.mixture.log_invtemplates[k,...,f])
                if res is None:
                    res = r1 + r2
                else:
                    res += r1 + r2

        return res
        
    @classmethod
    def load_from_dict(cls, d):
        try:
            num_mixtures = d['num_mixtures']
            patch_dict = PatchDictionary.load_from_dict(d['patch_dictionary'])
            obj = cls(num_mixtures, patch_dict)
            obj.mixture = ag.stats.BernoulliMixture.load_from_dict(d['mixture'])
            obj.settings = d['settings']
            return obj
        except KeyError, e:
            # TODO: Create a new exception for these kinds of problems
            raise Exception("Could not reconstruct class from dictionary. Missing '{0}'".format(e))

    def save_to_dict(self):
        d = {}
        d['num_mixtures'] = self.num_mixtures
        d['patch_dictionary'] = self.patch_dict.save_to_dict()
        d['mixture'] = self.mixture.save_to_dict()
        d['settings'] = self.settings
        return d
