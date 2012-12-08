
from __future__ import division
from patch_dictionary import PatchDictionary
import amitgroup as ag
import numpy as np
import scipy.signal
from saveable import Saveable

def mean_pooling(data, size):
    steps = tuple([data.shape[i]//size[i] for i in xrange(2)])
    if data.ndim == 3:
        output = np.zeros(steps + (data.shape[-1],))
    else:
        output = np.zeros(steps)

    print 'output', output.shape

    for i in xrange(steps[0]):
        for j in xrange(steps[1]):
            if data.ndim == 3: 
                output[i,j] = data[i*size[0]:(i+1)*size[0], j*size[1]:(j+1)*size[1]].mean(axis=0).mean(axis=0)
            else:
                output[i,j] = data[i*size[0]:(i+1)*size[0], j*size[1]:(j+1)*size[1]].mean()
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
        alpha_maps = []
        for i, filename in enumerate(images):
            ag.info(i, "Processing file", filename)
            edges, img = ag.features.bedges_from_image(filename, return_original=True, **self.patch_dict.bedges_settings())
            print 'edges.shape', edges.shape
        
            # Do some max pooling here.
        
            small = self.patch_dict.extract_pooled_parts(edges)

            print 'small.shape', small.shape

            if shape is None:
                shape = small.shape
                output = np.empty((len(images),) + small.shape)
                
            assert small.shape == shape, "Images must all be of the same size"
            output[i] = small 
            #all_slices.append(parts_list)
            #size = tuple([max(size[i], parts_list.shape[i]) for i in xrange(2)])
            alpha_maps.append(img[...,3])

        assert output is not None

        print 'output.shape', output.shape
    
        ag.info("Running mixture model in Detector")

        # Train mixture model OR SVM
        mixture = ag.stats.BernoulliMixture(self.num_mixtures, output)
        mixture.run_EM(1e-8)
        
        #self.templates = mixture.templates
        self.mixture = mixture

        # Pick out the support, by remixing the alpha channel
        self.support = self.mixture.remix(alpha_maps)

        # TODO: How to store the mixture model the best way?

    def response_map(self, image):
        """Retrieves log-likelihood response on 'image' (no scaling done)"""
        edges = ag.features.bedges_from_image(image, **self.patch_dict.bedges_settings())

        small = self.patch_dict.extract_pooled_parts(edges)
        print small.shape
        print 'templates shape', self.mixture.templates.shape

        num_mix = self.mixture.num_mix
        if 1:
            self.small_support = None
            for k in xrange(num_mix):
                #p = self.patch_dict.max_pooling(self.support[k])
                p = mean_pooling(self.support[k], self.patch_dict.settings['pooling_size'])
                import pylab as plt
                plt.imshow(p, interpolation='nearest'); plt.colorbar(); plt.show()
                if self.small_support is None:
                    self.small_support = np.zeros((num_mix,) + p.shape)
                self.small_support[k] = p
        
        if 0:
            import pylab as plt
            plt.imshow(self.small_support[2])
            plt.show()

        smallest = self.mixture.templates.min()
        for f in xrange(self.mixture.templates.shape[-1]):
            
            alpha = np.clip(self.small_support, 2*smallest, 1.0)
            self.mixture.templates[...,f] = self.mixture.templates[...,f] / alpha# + 0.5 * (1 - alpha)
            
        self.mixture.templates = np.clip(self.mixture.templates, smallest, 1-smallest)

        #self.mixture.templates = np.clip(self.mixture.templates, 0.5, 1.0)
        self.mixture.log_templates = np.log(self.mixture.templates)
        self.mixture.log_invtemplates = np.log(1.0-self.mixture.templates)

        if 1:
            import matplotlib.pylab as plt
            fig = plt.figure()
            ax = fig.add_subplot(111)
            l = plt.imshow(self.mixture.templates[2,...,170], vmin=0, vmax=1, cmap=plt.cm.RdBu, interpolation='nearest')
            plt.colorbar()
            from matplotlib.widgets import Slider
            
            axindex = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg='lightgoldenrodyellow')
            slider = Slider(axindex, 'Index', 0, 200-1)
            print 'templates', self.mixture.templates.shape
            
            def update(val):
                index = slider.val
                #l.set_data(small[...,index])
                l.set_data(self.mixture.templates[2,...,index])
                plt.draw()
            slider.on_changed(update)
            plt.show()

        
        print 'small.shape', small.shape
        print 'mixture.templates.shape', self.mixture.templates.shape

        res = None
        for k in [2]:#xrange(self.num_mixtures):
            kernel_mask = np.ones(self.small_support[k].shape, dtype=np.int8)
            #print small.shape, self.mixture.log_templates.shape
            if 0:
                #kernel_mask = self.small_support[k].astype(np.int8)
                kernel_mask = np.ones(self.mixture.templates.shape[1:-1], dtype=np.int8)
                from masked_convolve import masked_convolve

    
                r1 = masked_convolve(small, self.mixture.log_templates[k], kernel_mask)
                r2 = masked_convolve(1-small, self.mixture.log_invtemplates[k], kernel_mask)
                res = r1 + r2
            else:
                #self.mixture.templates = np.clip(self.mixture.templates, 0.5, 1-smallest)
                #self.mixture.templates = np.clip(self.mixture.templates, smallest, 0.5)
                self.mixture.log_templates = np.log(self.mixture.templates)
                self.mixture.log_invtemplates = np.log(1.0-self.mixture.templates)

                for f in xrange(small.shape[-1]):
                    smallf = small[...,f]
                    #print smallf.shape
                    sh = (self.mixture.log_templates.shape)
                    #print sh
                    bigger = ag.util.zeropad(smallf, (sh[1]//2, sh[2]//2))
                    #print bigger.shape
                    r1 = scipy.signal.convolve2d(bigger, self.mixture.log_templates[k,::-1,::-1,f], fillvalue=1, mode='valid')
                    r2 = scipy.signal.convolve2d(1-bigger, self.mixture.log_invtemplates[k,::-1,::-1,f], fillvalue=0, mode='valid')
                    if res is None:
                        res = r1 + r2
                    else:
                        res += r1 + r2

        return res, small

    def get_support_box_for_mix_comp(self, k):
        # Take the bounding box of the support, with a certain threshold.
        supp = self.support[k] 
        supp_axs = [supp.max(axis=i) for i in xrange(2)]

        th = 0.1 # threshold
        # Check first and last value of that threshold
        bb = [tuple(np.where(supp_axs[i] > th)[0][[0,-1]]) for i in xrange(2)]

        # This bb looks like [(x0, x1), (y0, y1)], when we want
        # it like [(x0, y0), (x1, y1)]. The following takes care of that
        return zip(*bb)
    
         

    @classmethod
    def load_from_dict(cls, d):
        try:
            num_mixtures = d['num_mixtures']
            patch_dict = PatchDictionary.load_from_dict(d['patch_dictionary'])
            obj = cls(num_mixtures, patch_dict)
            obj.mixture = ag.stats.BernoulliMixture.load_from_dict(d['mixture'])
            obj.settings = d['settings']
            obj.support = d['support']# > 0.2 # Todo
            print obj.support.shape 
            obj.small_support = None
            for k in xrange(num_mixtures):
                p = obj.patch_dict.max_pooling(obj.support[k])
                if obj.small_support is None:
                    obj.small_support = np.zeros((num_mixtures,) + p.shape)
                obj.small_support[k] = p
            return obj
        except KeyError, e:
            # TODO: Create a new exception for these kinds of problems
            raise Exception("Could not reconstruct class from dictionary. Missing '{0}'".format(e))

    def save_to_dict(self):
        d = {}
        d['num_mixtures'] = self.num_mixtures
        d['patch_dictionary'] = self.patch_dict.save_to_dict()
        d['mixture'] = self.mixture.save_to_dict()
        d['support'] = self.support
        d['settings'] = self.settings
        return d
