
from __future__ import division
from patch_dictionary import PatchDictionary
import amitgroup as ag
import numpy as np
import scipy.signal
from saveable import Saveable

# TODO: Eventually migrate
# TODO: Also, does it need to be general for ndim=3?
def mean_pooling(data, size):
    steps = tuple([data.shape[i]//size[i] for i in xrange(2)])
    if data.ndim == 3:
        output = np.zeros(steps + (data.shape[-1],))
    else:
        output = np.zeros(steps)

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
    def __init__(self, num_mixtures, patch_dict, settings={}):
        assert isinstance(patch_dict, PatchDictionary)
        self.patch_dict = patch_dict
        self.num_mixtures = num_mixtures
        self.mixture = None
        self.kernels = None
        self.log_kernels = None
        self.log_invkernels = None

        self.settings = {}
        self.settings['bounding_box_opacity_threshold'] = 0.1
        self.settings['min_probability'] = 0.05
        for k, v in settings.items():
            self.settings[k] = v
    
    def train_from_images(self, images):
        shape = None
        output = None
        alpha_maps = []
        for i, filename in enumerate(images):
            ag.info(i, "Processing file", filename)
            edges, img = ag.features.bedges_from_image(filename, return_original=True, **self.patch_dict.bedges_settings())
        
            # Extract the parts, with some pooling 
            small = self.patch_dict.extract_pooled_parts(edges)
            if shape is None:
                shape = small.shape
                output = np.empty((len(images),) + small.shape)
                
            assert small.shape == shape, "Images must all be of the same size, for now at least"
            output[i] = small 
            alpha_maps.append(img[...,3])

        ag.info("Running mixture model in Detector")

        # Train mixture model OR SVM
        mixture = ag.stats.BernoulliMixture(self.num_mixtures, output)
        mixture.run_EM(1e-8, self.settings['min_probability'])
        
        #self.templates = mixture.templates
        self.mixture = mixture

        # Pick out the support, by remixing the alpha channel
        self.support = self.mixture.remix(alpha_maps)

        self._preprocess()

    def _preprocess_pooled_support(self):
        """
        Pools the support to the same size as other types of pooling. However, we use
        mean pooling here, which might be different from the pooling we use for the
        features.
        """
        num_mix = self.mixture.num_mix
        self.small_support = None
        for k in xrange(num_mix):
            p = mean_pooling(self.support[k], self.patch_dict.settings['pooling_size'])
            if self.small_support is None:
                self.small_support = np.zeros((num_mix,) + p.shape)
            self.small_support[k] = p

    def _preprocess_kernels(self):
        # TODO: Change this to a 
        smallest = self.mixture.templates.min()
    
        # Now, we will extract the templates from the mixture model and
        # incorporate the support into it. We will call the result the kernel
        self.kernels = self.mixture.templates.copy()
        m0, m1 = self.kernels.min(), self.kernels.max()
        for f in xrange(self.mixture.templates.shape[-1]):
            
            # TODO: These are two competing ways. I need to confirm
            # the right way more scientifically. The second way produces    
            # better results (as far as I can see from the test images).
            #alpha = np.clip(self.small_support, 2.0*smallest, 1.0)
            #self.mixture.templates[...,f] = self.mixture.templates[...,f] / alpha
            #alpha = np.clip(self.small_support, smallest, 1.0-smallest)

            # The fact that they range from 0.05 to 0.95 causes some problem here.
            # We might even consider removing that constraint altogether, and enforce it
            # only after this step.
            alpha = self.small_support
            m = self.kernels[...,f].copy()
            m[m == m0] = 0.0
            m[m == m1] = 1.0
            self.kernels[...,f] = 0.5 * (1-alpha) + m#self.mixture.templates[...,f]# / alpha
            
        eps = self.settings['min_probability']
        self.kernels = np.clip(self.kernels, eps, 1-eps)

        #self.mixture.templates = np.clip(self.mixture.templates, 0.5, 1.0)
        self.log_kernels = np.log(self.kernels)
        self.log_invkernels = np.log(1.0-self.kernels)

    def response_map(self, image):
        """Retrieves log-likelihood response on 'image' (no scaling done)"""

        # Convert image to our feature space representation
        edges = ag.features.bedges_from_image(image, **self.patch_dict.bedges_settings())
        small = self.patch_dict.extract_pooled_parts(edges)

        res = None
        for k in [2]:#xrange(self.num_mixtures):
        #for k in xrange(num_mix):
            if 0:
                #kernel_mask = np.ones(self.small_support[k].shape, dtype=np.int8)
                #kernel_mask = self.small_support[k].astype(np.int8)
                kernel_mask = np.ones(self.mixture.templates.shape[1:-1], dtype=np.int8)
                from masked_convolve import masked_convolve

                r1 = masked_convolve(small, self.log_kernels[k], kernel_mask)
                r2 = masked_convolve(1-small, self.log_invkernels[k], kernel_mask)
                res = r1 + r2
            else:
                for f in xrange(small.shape[-1]):
                    # Pad the incoming image, so that the result will be the same size (this
                    # also allows us to detect objects partly cropped, even though it will be
                    # difficult - TODO: It might help if they get a score boost)
                    smallf = small[...,f]
                    sh = self.kernels.shape
                    bigger = ag.util.zeropad(smallf, (sh[1]//2, sh[2]//2))
                    r1 = scipy.signal.convolve2d(bigger, self.log_kernels[k,::-1,::-1,f], mode='valid')
                    r2 = scipy.signal.convolve2d(1-bigger, self.log_invkernels[k,::-1,::-1,f], mode='valid')
                    if res is None:
                        res = r1 + r2
                    else:
                        res += r1 + r2

        return res, small

    def get_support_box_for_mix_comp(self, k):
        """This returns a bounding box of the support for a given component"""
        # Take the bounding box of the support, with a certain threshold.
        supp = self.support[k] 
        supp_axs = [supp.max(axis=i) for i in xrange(2)]

        # TODO: Make into a setting
        th = self.settings['bounding_box_opacity_threshold'] # threshold
        # Check first and last value of that threshold
        bb = [tuple(np.where(supp_axs[i] > th)[0][[0,-1]]) for i in xrange(2)]

        # This bb looks like [(x0, x1), (y0, y1)], when we want
        # it like [(x0, y0), (x1, y1)]. The following takes care of that
        return zip(*bb)

    def _preprocess(self):
        """Pre-processes things"""
        self._preprocess_pooled_support()
        self._preprocess_kernels()

    # TODO: Very temporary, but could be useful code if tidied up
    def _temp__plot_feature_kernels(self):
        import matplotlib.pylab as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        l = plt.imshow(self.kernels[2,...,0], vmin=0, vmax=1, cmap=plt.cm.RdBu, interpolation='nearest')
        plt.colorbar()
        from matplotlib.widgets import Slider
        
        axindex = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg='lightgoldenrodyellow')
        slider = Slider(axindex, 'Index', 0, self.patch_dict.num_patches-1)
        
        def update(val):
            index = slider.val
            #l.set_data(small[...,index])
            l.set_data(self.mixture.templates[2,...,index])
            plt.draw()
        slider.on_changed(update)
        plt.show()


    @classmethod
    def load_from_dict(cls, d):
        try:
            num_mixtures = d['num_mixtures']
            patch_dict = PatchDictionary.load_from_dict(d['patch_dictionary'])
            obj = cls(num_mixtures, patch_dict)
            obj.mixture = ag.stats.BernoulliMixture.load_from_dict(d['mixture'])
            obj.settings = d['settings']
            obj.support = d['support']
            obj._preprocess()
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
