
from __future__ import division
import amitgroup as ag
import numpy as np
import scipy.signal
from saveable import Saveable
import gv

def probpad(data, padwidth, prob):
    data = np.asarray(data)
    shape = data.shape
    if isinstance(padwidth, int):
        padwidth = (padwidth,)*len(shape) 
        
    padded_shape = map(lambda ix: ix[1]+padwidth[ix[0]]*2, enumerate(shape))
    #new_data = np.ones(padded_shape, dtype=data.dtype) * (np.random.rand() < prob)
    new_data = np.random.random(padded_shape)
    for f in xrange(data.shape[-1]):
        new_data[...,f] = (new_data[...,f] < prob[f]).astype(data.dtype)
    #new_data = (np.random.random(padded_shape) < prob).astype(data.dtype)
    new_data[ [slice(w, -w) if w > 0 else slice(None) for w in padwidth] ] = data 
    return new_data

def pooling_size(data, size):
    return tuple([data.shape[i]//size[i] for i in xrange(2)])

# TODO: Eventually migrate 
def max_pooling(data, size):
    # TODO: Remove this later
    if size == (1, 1):
        return data
    steps = tuple([data.shape[i]//size[i] for i in xrange(2)])
    offset = tuple([data.shape[i]%size[i]//2 for i in xrange(2)])
    if data.ndim == 3:
        output = np.zeros(steps + (data.shape[-1],))
    else:
        output = np.zeros(steps)

    for i in xrange(steps[0]):
        for j in xrange(steps[1]):
            if data.ndim == 3: 
                output[i,j] = data[
                    offset[0]+i*size[0]:offset[0]+(i+1)*size[0], 
                    offset[1]+j*size[1]:offset[1]+(j+1)*size[1]
                ].max(axis=0).max(axis=0)
                #output[i,j] = data[offset[0]+i*size[0]:offset[1]+(i+1)*size[0], offset[1]+j*size[1]:offset[1]+(j+1)*size[1]].mean(axis=0).mean(axis=0)
                #output[i,j] = data[i*size[0],j*size[1]]
            else:
                output[i,j] = data[
                    offset[0]+i*size[0]:offset[0]+(i+1)*size[0], 
                    offset[1]+j*size[1]:offset[1]+(j+1)*size[1]].max()
                #output[i,j] = data[offset[0]+i*size[0]:offset[0]+(i+1)*size[0], offset[1]+j*size[1]:offset[1]+(j+1)*size[1]].mean()
                #output[i,j] = data[offset[0]+i*size[0],offset[1]+j*size[1]]
    return output


def subsample(data, size):
    offsets = [size[i]//2 for i in xrange(2)]
    return data[offsets[0]::size[0],offsets[1]::size[1]]
    

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
    An object detector representing a single class (although mutliple mixtures of that class).
        
    It uses the BinaryDescriptor as feature extractor, and then runs a mixture model on top of that.
    """
    def __init__(self, num_mixtures, descriptor, settings={}):
        assert isinstance(descriptor, gv.BinaryDescriptor)
        self.descriptor = descriptor 
        self.num_mixtures = num_mixtures
        self.mixture = None
        self.kernels = None
        self.log_kernels = None
        self.log_invkernels = None

        self.settings = {}
        self.settings['bounding_box_opacity_threshold'] = 0.1
        self.settings['min_probability'] = 0.05
        self.settings['pooling_size'] = (8, 8)
        self.settings.update(settings)
    
    def train_from_images(self, images):
        has_alpha = None#(img.shape[-1] == 4)
        
        shape = None
        output = None
        alpha_maps = []
        resize_to = self.settings.get('image_size')
        for i, img_obj in enumerate(images):
            if isinstance(img_obj, str):
                ag.info(i, "Processing file", img_obj)
                img = gv.img.load_image(img_obj)
                grayscale_img = img[...,:3].mean(axis=-1)
            else:
                ag.info(i, "Processing image of shape", img_obj.shape)
                grayscale_img = img_obj.mean(axis=-1)
    
            # Resize the image before extracting features
            if resize_to is not None and resize_to != grayscale_img.shape[:2]:
                img = gv.img.resize(img, resize_to)
                grayscale_img = gv.img.resize(grayscale_img, resize_to) 

            edges = self.extract_pooled_features(grayscale_img)
            #edges = self.descriptor.extract_features(grayscale_img)

            if has_alpha is None:
                has_alpha = (img.shape[-1] == 4)
        
            # Extract the parts, with some pooling 
            #small = self.descriptor.pool_features(edges)
            if shape is None:
                shape = edges.shape
                output = np.empty((len(images),) + edges.shape)
                
            assert edges.shape == shape, "Images must all be of the same size, for now at least"
            output[i] = edges 
            if has_alpha:
                alpha_maps.append(img[...,3])

        ag.info("Running mixture model in Detector")

        # Train mixture model OR SVM
        mixture = ag.stats.BernoulliMixture(self.num_mixtures, output.astype(np.uint8), float_type=np.float32)
        #mixture.run_EM(1e-6, self.settings['min_probability'])
        mixture.run_EM(1e-6, 1e-5)
        
        #self.templates = mixture.templates
        self.mixture = mixture

        # Pick out the support, by remixing the alpha channel
        if has_alpha:
            self.support = self.mixture.remix(alpha_maps)
        else:
            self.support = None#np.ones((self.num_mixtures,) + shape[:2])

        self._preprocess()

    def _preprocess_pooled_support(self):
        """
        Pools the support to the same size as other types of pooling. However, we use
        mean pooling here, which might be different from the pooling we use for the
        features.
        """

        self.small_support = None
        if self.support is not None:
            num_mix = self.mixture.num_mix
            for k in xrange(num_mix):
                p = mean_pooling(self.support[k], self.settings['pooling_size'])
                if self.small_support is None:
                    self.small_support = np.zeros((num_mix,) + p.shape)
                self.small_support[k] = p

    def _preprocess_kernels(self):
        self.kernels = self.mixture.templates.copy()

    def extract_pooled_features(self, image):
        #print image.shape
        edges = self.descriptor.extract_features(image, {'spread_radii': self.settings['spread_radii']})
        #small = max_pooling(edges, self.settings['pooling_size'])
        small = subsample(edges, self.settings['pooling_size'])
        return small

    def prepare_kernels(self, image, mixcomp, backTODO=None):
        # Convert image to our feature space representation
        edges = self.extract_pooled_features(image)
        #edges = self.descriptor.extract_features(image)

        num_features = edges.shape[-1]
        back = np.empty(num_features)
        for f in xrange(num_features):
            back[f] = edges[...,f].sum()
        back /= np.prod(edges.shape[:2])

        #back[:] = 0.05
        
        # Create kernels just for this case
        #large_kernels = self.large_kernels.copy()
        kernels = self.mixture.templates.copy()

        #import ipdb; ipdb.set_trace()
        # Support correction
        if self.support is not None:
            for f in xrange(num_features):
                # This is explained in writeups/cad-support/.
                kernels[mixcomp,...,f] += (1-self.small_support[mixcomp]) * back[f]
                #kernels[mixcomp,...,f] = 1 - (1 - kernels[mixcomp,...,f]) * (1 - back[f])**(1-self.small_support[mixcomp])
        

        # Pool kernels
        psize = self.settings['pooling_size']
        #for k in xrange(self.num_mixtures):          
        #    # TODO: MAX POOLING THE PROBABILITIES!? Doesn't make sense.
            #pooled = max_pooling(self.large_kernels[k], psize)
        #    if kernels is None:
        #        kernels = np.empty((self.num_mixtures,) + pooled.shape)
        #    kernels[k] = pooled

        eps = self.settings['min_probability']
        kernels = np.clip(kernels, eps, 1-eps)

        back_kernel = np.zeros(kernels.shape[1:]) 
        for f in xrange(edges.shape[-1]):
            back_kernel[...,f] = back[f]#edges[...,f].sum()

        back_kernel = np.clip(back_kernel, eps, 1-eps)

        self.kernels = kernels

        return back_kernel, kernels, edges

    def response_map(self, image, mixcomp):
        """Retrieves log-likelihood response on 'image' (no scaling done)"""

        back_kernel, kernels, edges = self.prepare_kernels(image, mixcomp)

        sh = kernels.shape
        padding = (sh[1]//2, sh[2]//2, 0)
        #bigger = ag.util.zeropad(edges, padding).astype(np.float64)
        bigger = probpad(edges, (sh[1]//2, sh[2]//2, 0), back_kernel[0,0])
        #bigger = ag.util.pad(edges, (sh[1]//2, sh[2]//2, 0), back_kernel[0,0])
        
        #bigger_minus_back = bigger.copy()

        for f in xrange(edges.shape[-1]):
            pass
            #bigger_minus_back[padding[0]:-padding[0],padding[1]:-padding[1],f] -= back_kernel[0,0,f] 
            #bigger_minus_back[padding[0]:-padding[0],padding[1]:-padding[1],f] -= kernels[mixcomp,...,f]


        res = None
        for k in [mixcomp]:#xrange(self.num_mixtures):
        #for k in xrange(self.mixture.num_mix):
            if 0:
                from masked_convolve import llh 
                print bigger.dtype, kernels[k].dtype
                res = llh(bigger, kernels[k].astype(np.float64)) 

                #summand = a**2 * back_kernel * (1 - back_kernel)
                #summand = a**2 * kernels[k] * (1 - kernels[k])
                #Z = np.sqrt(np.sum(summand))
                #print 'norm factor', Z
                #res /= Z
            
            elif 1:
                from masked_convolve import masked_convolve
                #r1 = masked_convolve(bigger, np.log(kernels[k]))
                #r2 = masked_convolve(1-bigger, np.log(1.0 - kernels[k]))
                #r3 = masked_convolve(1-bigger, -np.log(1.0 - back_kernel))
                #r4 = masked_convolve(bigger, -np.log(back_kernel))
                #print 'sizes', kernels[k].shape, back_kernel.shape
                a = np.log(kernels[k] * (1-back_kernel) / ((1-kernels[k]) * back_kernel))
                res = masked_convolve(bigger, a)
                
                # Subtract expected log likelihood
                res -= (back_kernel * a).sum()
                #res2 = (kernels[k] * a).sum()
                #import ipdb; ipdb.set_trace()
                #res -= res2

                # Normalize
                summand = a**2 * back_kernel * (1 - back_kernel)
                #summand = a**2 * kernels[k] * (1 - kernels[k])
                Z = np.sqrt(np.sum(summand))
                #print 'norm factor', Z
                res /= Z


            else:
                for f in xrange(small.shape[-1]):
                    # Pad the incoming image, so that the result will be the same size (this
                    # also allows us to detect objects partly cropped, even though it will be
                    # difficult - TODO: It might help if they get a score boost)
                    # TODO: Place outside of forloop (k) !
                    smallf = small[...,f]
                    sh = self.kernels.shape
                    bigger = ag.util.zeropad(smallf, (sh[1]//2, sh[2]//2))
                    # can also use fftconvolve
                    r1 = scipy.signal.convolve2d(bigger, self.log_kernels[k,::-1,::-1,f], mode='valid')
                    r2 = scipy.signal.convolve2d(1-bigger, self.log_invkernels[k,::-1,::-1,f], mode='valid')

                    r3 = bigger.sum() * np.log(1-self.back[f])
                    r4 = (1-bigger).sum() * np.log(self.back[f])
            

                    if res is None:
                        res = r1 + r2 + r3 + r4
                    else:
                        res += r1 + r2 + r3 + r4

        return res, edges 

    def unpooled_factor(self, side):
        return self.unpooled_kernel_side / side
    
    def factor(self, side):
        return self.kernel_side / side

    @property
    def unpooled_kernel_size(self):
        ps = self.settings['pooling_size']
        return (self.kernels.shape[1]*ps[0], self.kernels.shape[2]*ps[1])

    @property
    def kernel_size(self):
        return self.kernels.shape[1:3] #pooling_size(self.large_kernels[0], self.settings['pooling_size'])


    @property
    def unpooled_kernel_side(self):
        return max(self.unpooled_kernel_size)

    @property
    def kernel_side(self):
        # The side here is the maximum side of the kernel size 
        return max(self.kernel_size)

    def resize_and_detect(self, img, mixcomp, side=128):
        factor = self.unpooled_factor(side)
        img_resized = gv.img.resize_with_factor(img, factor)

        #print "calling response_map", img_resized.shape, mixcomp
        x, img_feat = self.response_map(img_resized, mixcomp)
        return x, img_feat, img_resized

    def detect_coarse_unfiltered_at_scale(self, img, side, mixcomp):
        x, small, img_resized = self.resize_and_detect(img, mixcomp, side)

        xx = x
        #th = xx.max() - 10.0
        th = 40.0
        top_th = 100.0
        #xx = (x - x.mean()) / x.std()
        #xx /= x.std()
        #xx /= #np.sqrt(np.sum(np.log(xx/(1-xx))**2 * xx * (1-xx))) 

        GET_ONE = False 
        if GET_ONE:
            th = xx.max() 

        bbs = []

        if 0:
            import pylab as plt
            plt.hist(xx.flatten(), 50)
            plt.show()

        #print 'feature activity:', small.sum() / np.prod(small.shape)
        #print 'x max', x.max()

        bb_bigger = (0.0, 0.0, img.shape[0], img.shape[1])

        for i in xrange(x.shape[0]):
            for j in xrange(x.shape[1]):
                if xx[i,j] >= th:
                    pooling_size = self.settings['pooling_size']
                    ix = i * pooling_size[0]
                    iy = j * pooling_size[1]
                    bb = self.bounding_box_at_pos((ix, iy), mixcomp)
                    # TODO: The above function should probably return in this coordinates
                    bb = tuple([bb[k] / self.unpooled_factor(side) for k in xrange(4)])
                    # Clip to bb_bigger 
                    bb = gv.bb.intersection(bb, bb_bigger)
                    score = xx[i,j]
                
                    conf = (score - th) / (top_th - th)
                    conf = np.clip(conf, 0, 1)
                    dbb = gv.bb.DetectionBB(score=score, box=bb, confidence=conf, scale=side)

                    if gv.bb.area(bb) > 0:
                        bbs.append(dbb)

        if GET_ONE:
            bbs.sort(reverse=True)
            bbs = bbs[:1]
            #print bbs[0]

        # Let's limit to five per level
        bbs = bbs[:5]
    
        return bbs, xx, small

    def detect_coarse(self, img, fileobj=None):
        bbs = []
        df = 0.05
        #df = 0.1
        #factors = np.arange(0.3, 1.0+0.01, df)
        factors = range(100, 401, 25)
        for factor in factors:
            for mixcomp in xrange(self.num_mixtures):
                bbsthis, _, _ = self.detect_coarse_unfiltered_at_scale(img, factor, mixcomp)
                bbs += bbsthis

        # Do NMS here
        final_bbs = self.nonmaximal_suppression2(bbs)
        
        # Mark corrects here
        if fileobj is not None:
            self.label_corrects(final_bbs, fileobj)
        return final_bbs

    def detect_coarse_single_component(self, img, mixcomp, fileobj=None):
        df = 0.05
        #df = 0.1
        #factors = np.arange(0.3, 1.0+0.01, df)
        factors = range(100, 401, 25)

        bbs = []
        for factor in factors:
            print "Running factor", factor
            bbsthis, _, _ = self.detect_coarse_unfiltered_at_scale(img, factor, mixcomp)
            print "found {0} bounding boxes".format(len(bbsthis))
            bbs += bbsthis
    
        # Do NMS here
        final_bbs = self.nonmaximal_suppression2(bbs)

        # Mark corrects here
        if fileobj is not None:
            self.label_corrects(final_bbs, fileobj)
        return final_bbs

    def nonmaximal_suppression2(self, bbs):
        # This one will respect scales a bit more
        bbs_sorted = sorted(bbs, reverse=True)

        overlap_threshold = 0.5

        #print 'bbs length', len(bbs_sorted)
        i = 1
        while i < len(bbs_sorted):
            # TODO: This can be vastly improved performance-wise
            for j in xrange(i):
                #print bb_area(bb_overlap(bbs[i].box, bbs[j].box))/bb_area(bbs[j].box)
                overlap = gv.bb.area(gv.bb.intersection(bbs_sorted[i].box, bbs_sorted[j].box))/gv.bb.area(bbs_sorted[j].box)
                if overlap > overlap_threshold and abs(bbs_sorted[i].scale - bbs_sorted[j].scale) <= 50: 
                    del bbs_sorted[i]
                    i -= 1
                    break

            i += 1
        #print 'bbs length', len(bbs_sorted)
        return bbs_sorted

    def nonmaximal_suppression(self, bbs):
        bbs_sorted = sorted(bbs, reverse=True)

        overlap_threshold = 0.5

        #print 'bbs length', len(bbs_sorted)
        i = 1
        while i < len(bbs_sorted):
            # TODO: This can be vastly improved performance-wise
            for j in xrange(0, i):
                #print bb_area(bb_overlap(bbs[i].box, bbs[j].box))/bb_area(bbs[j].box)
                if gv.bb.area(gv.bb.intersection(bbs_sorted[i].box, bbs_sorted[j].box))/gv.bb.area(bbs_sorted[j].box) > overlap_threshold: 
                    del bbs_sorted[i]
                    i -= 1
                    break

            i += 1
        #print 'bbs length', len(bbs_sorted)
        return bbs_sorted
    
    def bounding_box_for_mix_comp(self, k):
        """This returns a bounding box of the support for a given component"""

        # Take the bounding box of the support, with a certain threshold.
        #print self.support
        if self.support is not None:
            supp = self.support[k] 
            supp_axs = [supp.max(axis=1-i) for i in xrange(2)]

            # TODO: Make into a setting
            th = self.settings['bounding_box_opacity_threshold'] # threshold
            # Check first and last value of that threshold
            bb = [np.where(supp_axs[i] > th)[0][[0,-1]] for i in xrange(2)]

            # This bb looks like [(x0, x1), (y0, y1)], when we want it as (x0, y0, x1, y1)
            psize = self.settings['pooling_size']
            ret = (bb[0][0]/psize[0], bb[1][0]/psize[1], bb[0][1]/psize[0], bb[1][1]/psize[1])
            return ret
        else:
            #print '------------'
            #print self.support.shape
            #print self.kernels.shape
            #print self.small_support.shape
            return (0, 0, self.kernels.shape[1], self.kernels.shape[2])

    def bounding_box_at_pos(self, pos, mixcomp):
        supp_size = self.kernel_size 
        bb = self.bounding_box_for_mix_comp(mixcomp)

        # TODO: Is ps needed here? When should this be done?
        ps = self.settings['pooling_size']
        pos0 = [pos[i]-ps[i]*supp_size[i]//2 for i in xrange(2)]
        return (pos0[0]+bb[0]*ps[0],   
                pos0[1]+bb[1]*ps[1], 
                pos0[0]+bb[2]*ps[0], 
                pos0[1]+bb[3]*ps[1])


    def label_corrects(self, bbs, fileobj):
        used_bb = set([])
        tot = 0
        for bb2obj in bbs:
            bb2 = bb2obj.box
            best_score = None
            best_bbobj = None
            best_bb = None
            for bb1obj in fileobj.boxes: 
                bb1 = bb1obj.box
                if bb1 not in used_bb:
                    #print('union_area', gv.bb.union_area(bb1, bb2))
                    #print('intersection_area', gv.bb.area(gv.bb.intersection(bb1, bb2)))
                    #print('here', gv.bb.fraction_metric(bb1, bb2))
                    score = gv.bb.fraction_metric(bb1, bb2)
                    if score >= 0.5:
                        if best_score is None or score > best_score:
                            best_score = score
                            best_bbobj = bb2obj 
                            best_bb = bb1

            if best_bbobj is not None:
                best_bbobj.correct = True
                tot += 1
                used_bb.add(best_bb)

    def _preprocess(self):
        """Pre-processes things"""
        self._preprocess_pooled_support()
        self._preprocess_kernels()

    # TODO: Very temporary, but could be useful code if tidied up
    def _temp__plot_feature_kernels(self):
        assert 0, "This is broken since refactoring"
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
            descriptor_cls = gv.BinaryDescriptor.getclass(d['descriptor_name'])
            if descriptor_cls is None:
                raise Exception("The descriptor class {0} is not registered".format(d['descriptor_name'])) 
            descriptor = descriptor_cls.load_from_dict(d['descriptor'])
            obj = cls(num_mixtures, descriptor)
            obj.mixture = ag.stats.BernoulliMixture.load_from_dict(d['mixture'])
            obj.settings = d['settings']
            obj.support = d['support']

            # TODO: Pooling is not supported right now! 
            #obj.settings['pooling_size'] = (2, 2)
            obj._preprocess()
            return obj
        except KeyError, e:
            # TODO: Create a new exception for these kinds of problems
            raise Exception("Could not reconstruct class from dictionary. Missing '{0}'".format(e))

    def save_to_dict(self):
        d = {}
        d['num_mixtures'] = self.num_mixtures
        d['descriptor_name'] = self.descriptor.name
        d['descriptor'] = self.descriptor.save_to_dict()
        d['mixture'] = self.mixture.save_to_dict()
        d['support'] = self.support
        d['settings'] = self.settings
        return d
