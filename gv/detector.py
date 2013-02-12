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

def subsample_size(data, size):
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


def _subsample(data, size, skip_first_axis=False):
    offsets = [size[i]//2 for i in xrange(2)]
    if skip_first_axis:
        return data[:,offsets[0]::size[0],offsets[1]::size[1]]
    else:
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
        self.log_kernels = None
        self.log_invkernels = None
        self.scale_factor = 1.5

        self.settings = {}
        self.settings['bounding_box_opacity_threshold'] = 0.1
        self.settings['min_probability'] = 0.05
        self.settings['subsample_size'] = (8, 8)
        self.settings.update(settings)
    
    def train_from_images(self, images):
        has_alpha = None#(img.shape[-1] == 4)
        
        shape = None
        output = None
        unspread_output = None
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

            unspread_edges = self.extract_unspread_features(grayscale_img)
            edges = self.extract_spread_features(grayscale_img)
            #edges = self.descriptor.extract_features(grayscale_img)

            if has_alpha is None:
                has_alpha = (img.shape[-1] == 4)
        
            # Extract the parts, with some pooling 
            #small = self.descriptor.pool_features(edges)
            if shape is None:
                shape = edges.shape
                unspread_output = np.empty((len(images),) + unspread_edges.shape)
                output = np.empty((len(images),) + edges.shape)
                
            assert edges.shape == shape, "Images must all be of the same size, for now at least"
            output[i] = edges 
            unspread_output[i] = unspread_edges
            if has_alpha:
                alpha_maps.append(img[...,3])

        ag.info("Running mixture model in Detector")

        # Train mixture model OR SVM
        mixture = ag.stats.BernoulliMixture(self.num_mixtures, output.astype(np.uint8), float_type=np.float32)
        #mixture.run_EM(1e-6, self.settings['min_probability'])
        mixture.run_EM(1e-6, 1e-5)
        
        #self.templates = mixture.templates
        self.mixture = mixture

        # Now create our unspread kernels
        self.kernel_templates = self.mixture.remix(unspread_output) 

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

        if 0:
            self.small_support = None
            if self.support is not None:
                num_mix = self.mixture.num_mix
                for k in xrange(num_mix):
                    p = mean_pooling(self.support[k], self.settings['subsample_size'])
                    if self.small_support is None:
                        self.small_support = np.zeros((num_mix,) + p.shape)
                    self.small_support[k] = p

    def _preprocess_kernels(self):
        pass#self.kernels = self.mixture.templates.copy()

    def extract_unspread_features(self, image):
        edges = self.descriptor.extract_features(image, {'spread_radii': (0, 0)})
        return edges

    def extract_spread_features(self, image):
        edges = self.descriptor.extract_features(image, {'spread_radii': self.settings['spread_radii']})
        return edges 

    @property
    def unpooled_kernel_size(self):
        return (self.kernel_templates.shape[1], self.kernel_templates.shape[2])

    @property
    def unpooled_kernel_side(self):
        return max(self.unpooled_kernel_size)


    def background_model(self, edges):
        num_features = edges.shape[-1]

        back = np.empty(num_features)
        for f in xrange(num_features):
            back[f] = edges[...,f].sum()
        back /= np.prod(edges.shape[:2])

        if 0:
            #import pylab as plt
            K = 2 
            flat_edges = edges.reshape((np.prod(edges.shape[:2]),-1))
            backmodel = ag.stats.BernoulliMixture(K, flat_edges)
            backmodel.run_EM(1e-8, 0.05)

            #import ipdb; ipdb.set_trace()

            aa = np.argmax(backmodel.affinities.reshape(edges.shape[:2]+(-1,)), axis=-1)
            if 0:
                plt.figure(figsize=(8, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
                plt.subplot(1, 2, 2)
                plt.imshow(aa)
                plt.show()

            if 0:
                for i in xrange(K):
                    plt.subplot(2, 2, 1+i)
                    plt.plot(backmodel.templates[i], drawstyle='steps')
                    plt.ylim((0, 1))
                    
                plt.show()

            # Choose the loudest
            back_i = np.argmax(backmodel.templates.sum(axis=-1))
            print 'back i', back_i
            back = backmodel.templates[back_i]

        eps = self.settings['min_probability']
        back = np.clip(back, eps, 1-eps)
    
        return back

    def subsample(self, edges):
        return _subsample(edges, self.settings['subsample_size'])

    def prepare_kernels(self, back):
        num_features = back.size

        kernels = self.kernel_templates.copy()

        if 1:
            spread_N = 3
            nospread_back = 1 - (1 - back)**(1/(2*spread_N+1)**2)

            for mixcomp in xrange(self.num_mixtures):
                # Fix kernels
                krn = kernels[mixcomp].copy()

                #for f in xrange(num_features):
                for i in xrange(krn.shape[0]):
                    for j in xrange(krn.shape[1]):
                        p = np.ones(num_features)
                        for u in xrange(-3, 4):
                            for v in xrange(-3, 4):
                                if 0 <= i+u < krn.shape[0] and \
                                   0 <= j+v < krn.shape[1]:
                                    p *= (1 - kernels[mixcomp,i+u,j+v]) - nospread_back * (1-self.support[mixcomp,i+u,j+v])
                                else:
                                    p *= (1 - nospread_back)

                        krn[i,j] = 1 - p

                f0 = 0
            
                if 0:
                    import pylab as plt
                    plt.subplot(1, 2, 1)
                    plt.imshow(kernels[mixcomp,...,f0], interpolation='nearest')
                    plt.colorbar()
                    plt.subplot(1, 2, 2)
                    plt.imshow(krn[...,f0], interpolation='nearest')
                    plt.colorbar()
                    plt.show()

                kernels[mixcomp] = krn

        #import ipdb; ipdb.set_trace()
        # Support correction
        for mixcomp in xrange(self.num_mixtures):
            if self.support is not None:
                for f in xrange(num_features):
                    # This is explained in writeups/cad-support/.
                    pass
                    #kernels[mixcomp,...,f] += (1-self.support[mixcomp]) * back[f]
                    #kernels[mixcomp,...,f] = 1 - (1 - kernels[mixcomp,...,f]) * (1 - back[f])**(1-self.support[mixcomp])
        

        # Subsample kernels
        psize = self.settings['subsample_size']
        sub_kernels = _subsample(kernels, psize, skip_first_axis=True)

        eps = self.settings['min_probability']
        sub_kernels = np.clip(sub_kernels, eps, 1-eps)

        return sub_kernels

    def detect_coarse_single_factor(self, img, factor, mixcomp):
        from skimage.transform import pyramid_reduce
        img_resized = pyramid_reduce(img, downscale=factor)
    
        up_feats = self.extract_unspread_features(img_resized)
        bkg = self.background_model(up_feats)
        feats = self.subsample(up_feats) 

        # Prepare kernel
        sub_kernels = self.prepare_kernels(bkg)

        bbs, resmap = self.detect_coarse_at_factor(feats, sub_kernels, bkg, factor, mixcomp)

        # Do NMS here
        final_bbs = self.nonmaximal_suppression(bbs)

        return final_bbs, resmap

    def detect_coarse(self, img, fileobj=None):
        # Build image pyramid
        from skimage.transform import pyramid_gaussian 
        min_size = 75
        min_factor = min_size / self.unpooled_kernel_side

        max_size = 450
        max_factor = max_size / self.unpooled_kernel_side

        num_levels = 2
        factors = []
        skips = 0
        for i in xrange(1000):
            factor = self.scale_factor**i
            if factor > max_factor:
                break
            if factor >= min_factor:
                factors.append(factor) 
            else:
                skips += 1
        num_levels = len(factors) + skips

        pyramid = list(pyramid_gaussian(img, max_layer=num_levels, downscale=self.scale_factor))[skips:]

        # Filter out levels that are below minimum scale

        # Prepare each level 
        edge_pyramid = map(self.extract_unspread_features, pyramid)
        bkg_pyramid = map(self.background_model, edge_pyramid)
        small_pyramid = map(self.subsample, edge_pyramid) 

        bbs = []
        for i, factor in enumerate(factors):
            # Prepare the kernel for this mixture component
            sub_kernels = self.prepare_kernels(bkg_pyramid[i])

            for mixcomp in xrange(self.num_mixtures):
                bbsthis, _ = self.detect_coarse_at_factor(small_pyramid[i], sub_kernels, bkg_pyramid[i], factor, mixcomp)
                bbs += bbsthis

        # Do NMS here
        final_bbs = self.nonmaximal_suppression(bbs)
        
        # Mark corrects here
        if fileobj is not None:
            self.label_corrects(final_bbs, fileobj)

        return final_bbs

    def detect_coarse_at_factor(self, sub_feats, sub_kernels, back, factor, mixcomp):
        # Get background level
        
        resmap = self.response_map(sub_feats, sub_kernels, back, mixcomp)

        th = 30.0  
        top_th = 200.0
        bbs = []
        
        psize = self.settings['subsample_size']
        agg_factors = tuple([psize[i] * factor for i in xrange(2)])
        bb_bigger = (0.0, 0.0, sub_feats.shape[0] * agg_factors[0], sub_feats.shape[1] * agg_factors[1])
        for i in xrange(resmap.shape[0]):
            for j in xrange(resmap.shape[1]):
                score = resmap[i,j]
                if score >= th:
                    ix = i * psize[0]
                    iy = j * psize[1]

                    #import ipdb; ipdb.set_trace()
                    i_corner = i-sub_kernels.shape[1]//2
                    j_corner = j-sub_kernels.shape[2]//2

                    obj_bb = self.bounding_box_for_mix_comp(mixcomp)
                    bb = [(i_corner + obj_bb[0]) * agg_factors[0],
                          (j_corner + obj_bb[1]) * agg_factors[1],
                          (i_corner + obj_bb[2]) * agg_factors[0],
                          (j_corner + obj_bb[3]) * agg_factors[1],
                    ]

                    # Clip to bb_bigger 
                    bb = gv.bb.intersection(bb, bb_bigger)
                
                    conf = (score - th) / (top_th - th)
                    conf = np.clip(conf, 0, 1)
                    dbb = gv.bb.DetectionBB(score=score, box=bb, confidence=conf, scale=factor)

                    if gv.bb.area(bb) > 0:
                        bbs.append(dbb)

        # Let's limit to five per level
        bbs_sorted = self.nonmaximal_suppression(bbs)
        bbs_sorted = bbs_sorted[:5]

        return bbs_sorted, resmap

    def response_map(self, sub_feats, sub_kernels, back, mixcomp):
        sh = sub_kernels.shape
        padding = (sh[1]//2, sh[2]//2, 0)
        bigger = ag.util.zeropad(sub_feats, padding).astype(np.float64)
        #bigger = probpad(edges, (sh[1]//2, sh[2]//2, 0), back_kernel[0,0])
        #bigger = ag.util.pad(edges, (sh[1]//2, sh[2]//2, 0), back_kernel[0,0])
        
        #bigger_minus_back = bigger.copy()

        #for f in xrange(edges.shape[-1]):
        #    pass
            #bigger_minus_back[padding[0]:-padding[0],padding[1]:-padding[1],f] -= back_kernel[0,0,f] 
            #bigger_minus_back[padding[0]:-padding[0],padding[1]:-padding[1],f] -= kernels[mixcomp,...,f]

        res = None

        if 0:
            from masked_convolve import llh 
            print bigger.dtype, kernels[mixcomp].dtype
            res = llh(bigger, kernels[mixcomp].astype(np.float64)) 

            #summand = a**2 * back_kernel * (1 - back_kernel)
            #summand = a**2 * kernels[k] * (1 - kernels[k])
            #Z = np.sqrt(np.sum(summand))
            #print 'norm factor', Z
            #res /= Z
        
        a = np.log(sub_kernels[mixcomp] / (1-sub_kernels[mixcomp]) * ((1-back) / back))

        # With larger kernels, the fftconvolve is much faster
        if 1:
            from fast import multifeature_correlate2d 
            res = multifeature_correlate2d(bigger, a)
        else:
            areversed = a[::-1,::-1]
            biggo = np.rollaxis(bigger, axis=-1)
            arro = np.rollaxis(areversed, axis=-1)

            #import pdb; pdb.set_trace()
    
            for f in xrange(sub_feats.shape[-1]):
                r1 = scipy.signal.fftconvolve(biggo[f], arro[f], mode='valid')
                if res is None:
                    res = r1
                else:
                    res += r1
        
        # Subtract expected log likelihood
        res -= (a * back).sum()

        # Standardize 
        summand = a**2 * (back * (1 - back))
        #summand = a**2 * kernels[k] * (1 - kernels[k])
        Z = np.sqrt(np.sum(summand))
        #print 'norm factor', Z
        res /= Z
        
        return res

    def nonmaximal_suppression(self, bbs):
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
                print bbs_sorted[i].scale, bbs_sorted[j].scale
                if overlap > overlap_threshold and 1/self.scale_factor-0.01 <= (bbs_sorted[i].scale / bbs_sorted[j].scale) <= self.scale_factor+0.01: 
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
            psize = self.settings['subsample_size']
            ret = (bb[0][0]/psize[0], bb[1][0]/psize[1], bb[0][1]/psize[0], bb[1][1]/psize[1])
            return ret
        else:
            #print '------------'
            return (0, 0, self.kernel_templates.shape[1], self.kernel_templates.shape[2])

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
        l = plt.imshow(self.kernel_templates[2,...,0], vmin=0, vmax=1, cmap=plt.cm.RdBu, interpolation='nearest')
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
            obj.kernel_templates = d['kernel_templates']
            obj.support = d['support']
            # TODO: VERY TEMPORARY!
            obj.settings['subsample_size'] = (4, 4)

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
        d['kernel_templates'] = self.kernel_templates
        d['support'] = self.support
        d['settings'] = self.settings
        return d
