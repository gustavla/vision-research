from __future__ import division

import amitgroup as ag
import numpy as np
import scipy.signal
from .saveable import Saveable
import gv
import sys
from copy import deepcopy

# TODO: Build into train_basis_...
#cad_kernels = np.load('cad_kernel.npy')

def offset_img(img, off):
    sh = img.shape
    if sh == (0, 0):
        return img
    else:
        x = np.zeros(sh)
        x[max(off[0], 0):min(sh[0]+off[0], sh[0]), \
          max(off[1], 0):min(sh[1]+off[1], sh[1])] = \
            img[max(-off[0], 0):min(sh[0]-off[0], sh[0]), \
                max(-off[1], 0):min(sh[1]-off[1], sh[1])]
        return x

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
        self.kernel_basis = None
        self.kernel_basis_samples = None
        self.kernel_templates = None
        self.kernel_sizes = None
        self.support = None
        self.fixed_bkg = None
        self.fixed_spread_bkg = None

        self.use_alpha = None

        self.settings = {}
        self.settings['scale_factor'] = np.sqrt(2)
        self.settings['bounding_box_opacity_threshold'] = 0.1
        self.settings['min_probability'] = 0.05
        self.settings['subsample_size'] = (8, 8)
        self.settings['train_unspread'] = True
        self.settings['min_size'] = 75
        self.settings['max_size'] = 450
        self.settings.update(settings)
    
    def copy(self):
        return deepcopy(self)

    @property
    def train_unspread(self):
        return self.settings['train_unspread']

    @property
    def num_features(self):
        return self.descriptor.num_features
    
    @property
    def use_basis(self):
        return self.kernel_basis is not None

    def load_img(self, images, offsets=None):
        resize_to = self.settings.get('image_size')
        for i, img_obj in enumerate(images):
            if isinstance(img_obj, str):
                #print("Image file name", img_obj)
                img = gv.img.load_image(img_obj)
            grayscale_img = gv.img.asgray(img)

            # Resize the image before extracting features
            if resize_to is not None and resize_to != grayscale_img.shape[:2]:
                img = gv.img.resize(img, resize_to)
                grayscale_img = gv.img.resize(grayscale_img, resize_to) 

            # Offset the image
            if offsets is not None:
                grayscale_img = offset_img(grayscale_img, offsets[i])
                img = offset_img(img, offsets[i])

            # Now, binarize the support in a clever way (notice that we have to adjust for pre-multiplied alpha)
            alpha = (img[...,3] > 0.2)

            eps = sys.float_info.epsilon
            imrgb = (img[...,:3]+eps)/(img[...,3:4]+eps)
            
            new_img = imrgb * alpha.reshape(alpha.shape+(1,))

            new_grayscale_img = new_img[...,:3].mean(axis=-1)

            yield i, grayscale_img, img, alpha

    def gen_img(self, images, actual=False):
        for i, grayscale_img, img, alpha in self.load_img(images):
            final_edges = self.extract_spread_features(grayscale_img)
            #final_edges = self.subsample(final_edges)
            yield final_edges

    def train_from_images(self, images):
        self.orig_kernel_size = None

        mixture, kernel_templates, kernel_sizes, support = self._train(images)

        self.mixture = mixture
        self.kernel_templates = kernel_templates
        self.kernel_sizes = kernel_sizes
        self.support = support
            
        self._preprocess()

    def _train(self, images, offsets=None):
        self.use_alpha = None
        
        real_shape = None
        shape = None
        output = None
        final_output = None
        alpha_maps = None 
        sparse = False # TODO: Change
        build_sparse = True and sparse
        feats = None

        # TODO: Remove
        orig_output = None
        psize = self.settings['subsample_size']

        for i, grayscale_img, img, alpha in self.load_img(images, offsets):
            ag.info(i, "Processing image", i)
            if self.use_alpha is None:
                self.use_alpha = (img.ndim == 3 and img.shape[-1] == 4)
                #if self.use_alpha:
                alpha_maps = np.empty((len(images),) + img.shape[:2], dtype=np.uint8)

            #if self.use_alpha:
            a = (img[...,3] > 0.05).astype(np.uint8)
            alpha_maps[i] = a

            edges_nonflat = self.extract_spread_features(grayscale_img)
            #edges_nonflat = gv.sub.subsample(orig_edges, psize)
            if shape is None:
                shape = edges_nonflat.shape

            edges = edges_nonflat.ravel()

            if self.orig_kernel_size is None:
                self.orig_kernel_size = (img.shape[0], img.shape[1])
        
            # Extract the parts, with some pooling 
            if output is None:
                if sparse:
                    if build_sparse:
                        output = scipy.sparse.dok_matrix((len(images),) + edges.shape, dtype=np.uint8)
                    else:
                        output = np.zeros((len(images),) + edges.shape, dtype=np.uint8)
                else:
                    output = np.empty((len(images),) + edges.shape, dtype=np.uint8)

                #orig_output = np.empty((len(images),) + orig_edges.shape, dtype=np.uint8)
            
            #orig_output[i] = orig_edges
                
            if build_sparse:
                for j in np.where(edges==1):
                    output[i,j] = 1
            else:
                output[i] = edges

        ag.info("Running mixture model in Detector")

        if output is None:
            raise Exception("Found no training images")

        if build_sparse:
            output = output.tocsr()
        elif sparse:
            output = scipy.sparse.csr_matrix(output)
        else:
            output = np.asmatrix(output)


        seed = self.settings.get('init_seed', 0)

        # Train mixture model OR SVM
        mixture = ag.stats.BernoulliMixture(self.num_mixtures, output, float_type=np.float32, init_seed=seed)

        minp = 0.05
        mixture.run_EM(1e-10, minp)

        #mixture.templates = np.empty(0)

        # Now create our unspread kernels
        # Remix it - this iterable will produce each object and then throw it away,
        # so that we can remix without having to ever keep all mixing data in memory at once

        kernel_templates = np.clip(mixture.templates.reshape((self.num_mixtures,) + shape), 1e-5, 1-1e-5)
        kernel_sizes = [self.settings['image_size']] * self.num_mixtures

        #support = None
        if self.use_alpha:
            support = mixture.remix(alpha_maps).astype(np.float32) 
        else:
            support = None

        self.support = support
        if 0:
            kernel_templates = np.clip(mixture.remix_iterable(self.gen_img(images)), 1e-5, 1-1e-5)
            if 0:
                # Pick out the support, by remixing the alpha channel
                if self.use_alpha: #TODO: Temp2
                    support = mixture.remix(alpha_maps).astype(np.float32)
                    # TODO: Temporary fix
                    self.full_support = support
                    support = support[:,6:-6,6:-6]
                else:
                    support = None#np.ones((self.num_mixtures,) + shape[:2])

                # TODO: Figure this out.
                self.support = support


        # Determine the log likelihood of the training data
        testing_type = self.settings.get('testing_type')
        fix_bkg = self.settings.get('fixed_bkg')
        if self.settings.get('bkg_type') == 'from-file':
            self.fixed_bkg = np.load(self.settings['fixed_bkg_file'])
            self.fixed_spread_bkg = np.load(self.settings['fixed_spread_bkg_file'])

        #fixed_bkg_file = self.settings.get('fixed_bkg_file')

        radii = self.settings['spread_radii']
        if testing_type == 'fixed':
            #L = len(self.settings['levels'])
            self.fixed_train_mean = np.zeros(self.num_mixtures)
            self.fixed_train_std = np.zeros(self.num_mixtures)

            #for i, (sub, spread) in enumerate(self.settings['levels']):

            psize = self.settings['subsample_size']
            radii = self.settings['spread_radii']

            if 0:
                orig_output = None
                # Get images with the right amount of spreading
                for j, grayscale_img, img, alpha in self.load_img(images, offsets):
                    orig_edges = self.extract_spread_features(grayscale_img, settings=dict(spread_radii=radii))
                    if orig_output is None:
                        orig_output = np.empty((len(images),) + orig_edges.shape, dtype=np.uint8)
                    orig_output[j] = orig_edges
                        #orig_edges = self.extract_spread_features(grayscale_img)
                
            self.kernel_templates = kernel_templates

            #bkg = 1 - (1 - fix_bkg)**((2 * radii[0] + 1) * (2 * radii[1] + 1))
            unspread_bkg = self.bkg_model(None, spread=False)
            spread_bkg = self.bkg_model(None, spread=True)
            #bkg = 0.05

            # TODO: This gives a spread background!
            kernels = self.prepare_kernels(unspread_bkg, settings=dict(spread_radii=radii, subsample_size=psize))

            #sub_output = gv.sub.subsample(orig_output, psize, skip_first_axis=True)

            #import pylab as plt
            #plt.imshow(kernels[0].sum(axis=-1), interpolation='nearest')
            #plt.show()
            X = np.asarray(output).reshape((-1,) + shape) #sub_output.reshape((sub_output.shape[0], -1))
            llhs = [[] for i in xrange(self.num_mixtures)] 

            comps = mixture.mixture_components()
            for i, Xi in enumerate(X):
                mixcomp = comps[i]
                a = np.log(kernels[mixcomp]/(1-kernels[mixcomp]) * ((1-spread_bkg)/spread_bkg))
                llh = np.sum(Xi * a)
                llhs[mixcomp].append(llh)

            self.fixed_train_mean = np.asarray([np.mean(llhs[k]) for k in xrange(self.num_mixtures)]) 
            self.fixed_train_std = np.asarray([np.std(llhs[k]) for k in xrange(self.num_mixtures)])

            if 0:
                #print('sub_output', sub_output.shape)
                theta = kernels.reshape((kernels.shape[0], -1))
                X = np.asarray(output) #sub_output.reshape((sub_output.shape[0], -1))

                llhs = np.dot(X, np.log(theta/(1-theta) * ((1-bkg)/bkg)).T)
                #C = np.log((1-theta)/(1-bkg)).sum(axis=1)
                #llhs += C
                
                lrt = llhs.max(axis=1)

                self.fixed_train_mean[i] = lrt.mean()
                self.fixed_train_std[i] = lrt.std()
    
        return mixture, kernel_templates, kernel_sizes, support

    def extract_unspread_features(self, image):
        edges = self.descriptor.extract_features(image, dict(spread_radii=(0, 0), crop_border=self.settings.get('crop_border')))
        return edges

    def extract_spread_features(self, image):
        edges = self.descriptor.extract_features(image, dict(spread_radii=self.settings['spread_radii'], subsample_size=self.settings['subsample_size'], crop_border=self.settings.get('crop_border')))
        return edges 

    @property
    def unpooled_kernel_size(self):
        return self.kernel_templates[0].shape[:2]

    @property
    def unpooled_kernel_side(self):
        return max(self.unpooled_kernel_size)

    def bkg_model(self, edges, spread=False, location=None):
        """
        Returns unspread background model in three different ways:

        * As a (num_features,) long vector, valid for the 
          entire image

        * As a (size0, size1, num_features) with a separate 
          background model for each pixel

        * As a (obj_size0, obj_size1, num_features) for a 
          separate background for each pixel inside the
          object. If location = True, then this format will
          be used.

        """
        bkg_type = self.settings.get('bkg_type')

        #return np.load('spread_bkg.npy')
    
        if bkg_type == 'constant':
            bkg_value = self.settings['fixed_bkg']
            return np.ones(self.num_features) * bkg_value 

        elif bkg_type == 'corner':
            assert not self.settings.get('train_unspread')
            if spread:
                return self.kernel_templates[0][0,0]
            else:
                return None

        elif bkg_type == 'from-file':
            if spread:
                return self.fixed_spread_bkg
            else:
                return self.fixed_bkg

        elif bkg_type == 'per-image-average':
            bkg = edges.reshape((-1, self.num_features)).mean(axis=0)
            # TODO: min_probability is too high here
            eps = 1e-10
            bkg = np.clip(bkg, eps, 1 - eps)
            return bkg

        elif bkg_type == 'smoothed':
            pass

        else:
            raise ValueError("Specified background model not available")

    #def subsample(self, edges):
        #return gv.sub.subsample(edges, self.settings['subsample_size'])

    def prepare_kernels(self, unspread_bkg, settings={}):
        sett = self.settings.copy()
        sett.update(settings) 

        if sett.get('kernel_ready'):
            return self.kernel_templates 

        if not self.use_basis:
            kernels = deepcopy(self.kernel_templates)

        eps = sett['min_probability']
        psize = sett['subsample_size']

        if self.train_unspread:
            # TODO: This does not handle irregular-sized kernel_templates objects!

            radii = sett['spread_radii']
            #neighborhood_area = ((2*radii[0]+1)*(2*radii[1]+1))

            if self.use_basis:
                #global cad_kernels
                import time
                start = time.time()
                a = 1 - unspread_bkg.sum()
                bkg_categorical = np.concatenate(([a], unspread_bkg))

                C = self.kernel_basis * np.expand_dims(bkg_categorical, -1)
                kernels = C.sum(axis=-2) / self.kernel_basis_samples.reshape((-1,) + (1,)*(C.ndim-2))

                kernels = np.clip(kernels, 1e-5, 1-1e-5)
                end = time.time()
                print (end - start) * 1000, 'ms'

            #unspread_bkg = 1 - (1 - bkg)**(1/neighborhood_area)
            #unspread_bkg = 1 - (1 - bkg)**50
            unspread_bkg = np.clip(unspread_bkg, 1e-5, 1-1e-5)
        
            aa_log = [ag.util.multipad(np.log(1 - kernel), (radii[0], radii[1], 0), np.log(1-unspread_bkg)) for kernel in kernels]

            integral_aa_log = [aa_log_i.cumsum(1).cumsum(2) for aa_log_i in aa_log]

            offsets = gv.sub.subsample_offset(kernels[0], psize)

            # Fix kernels
            istep = 2*radii[0]
            jstep = 2*radii[1]
            for mixcomp in xrange(self.num_mixtures):
                sh = kernels[mixcomp].shape[:2]
                # Note, we are going in strides of psize, given a certain offset, since
                # we will be subsampling anyway, so we don't need to do the rest.
                for i in xrange(offsets[0], sh[0], psize[0]):
                    for j in xrange(offsets[1], sh[1], psize[1]):
                        p = gv.img.integrate(integral_aa_log[mixcomp], i, j, i+istep, j+jstep)
                        kernels[mixcomp][i,j] = 1 - np.exp(p)

            

            # Subsample kernels
            sub_kernels = [gv.sub.subsample(kernel, psize) for kernel in kernels]
        else:
            sub_kernels = kernels

            if self.use_basis:
                a = 1 - unspread_bkg.sum()

                C = self.kernel_basis * np.expand_dims(unspread_bkg, -1)
                kernels = a * cad_kernels + C.sum(axis=-2) / self.kernel_basis_samples
                

        for i in xrange(self.num_mixtures):
            sub_kernels[i] = np.clip(sub_kernels[i], eps, 1-eps)

        K = self.settings.get('quantize_bins')
        if K is not None:
            assert 0, "Does not work with different size kernels"
            sub_kernels = np.round(1 + sub_kernels * (K - 2)) / K


        return sub_kernels

    def detect_coarse_single_factor(self, img, factor, mixcomp, img_id=0):
        """
        TODO: Experimental changes under way!
        """
        

        img_resized = gv.img.resize_with_factor_new(gv.img.asgray(img), 1/factor) 

        last_resmap = None

        psize = self.settings['subsample_size']
        radii = self.settings['spread_radii']
        cb = self.settings.get('crop_border')

        #spread_feats = self.extract_spread_features(img_resized)
        spread_feats = self.descriptor.extract_features(img_resized, dict(spread_radii=radii, subsample_size=psize))
        unspread_feats = self.descriptor.extract_features(img_resized, dict(spread_radii=(0, 0), subsample_size=psize, crop_border=cb))

        # TODO: Avoid the edge for the background model
        spread_bkg = self.bkg_model(spread_feats, spread=True)
        unspread_bkg = self.bkg_model(unspread_feats, spread=False)
        #unspread_bkg = np.load('bkg.npy')
        #spread_bkg = 1 - (1 - unspread_bkg)**25
        #spread_bkg = np.load('spread_bkg.npy')

        #feats = gv.sub.subsample(spread_feats, psize) 
        sub_kernels = self.prepare_kernels(unspread_bkg, settings=dict(spread_radii=radii, subsample_size=psize))

        bbs, resmap = self.detect_coarse_at_factor(spread_feats, sub_kernels, spread_bkg, factor, mixcomp)

        final_bbs = bbs

        return final_bbs, resmap, spread_feats, img_resized

    def calc_score(self, img, factor, bbobj, score=0):
        llhs = score
    
        i0, j0 = bbobj.score0, bbobj.score1

        # TODO: Temporary
        img_resized = gv.img.resize_with_factor_new(img, factor)
        factor = 1.
        mixcomp = 0

        psize = self.settings['subsample_size']
        radii = self.settings['spread_radii']

        feats = self.extract_spread_features(img_resized)

        # Last psize
        d0, d1 = (14, 44) 

        pad = 50 

        unspread_feats = extract2(img_resized)
        #unspread_feats_pad = ag.util.zeropad(unspread_feats, (pad, pad, 0))
        #unspread_feats0 = unspread_feats_pad[-10 + pad+i0-d0//2:10+ pad+i0-d0//2+d0, -10 + pad+j0-d1//2:10 + pad+j0-d1//2+d1]

        #bkg = self.bkg_model(unspread_feats0)
        unspread_bkg = self.bkg_model(unspread_feats, spread=False)

        #feats = gv.sub.subsample(up_feats, psize) 
        feats_pad = ag.util.zeropad(feats, (pad, pad, 0))
        feats0 = feats_pad[pad+i0-d0//2:pad+i0-d0//2+d0, pad+j0-d1//2:pad+j0-d1//2+d1]
        if 0:
            sub_kernels = self.prepare_kernels(unspread_bkg, settings=dict(spread_radii=radii, subsample_size=psize))

            neighborhood_area = ((2*radii[0]+1)*(2*radii[1]+1))
            # TODO: Don't do this anymore
            spread_back = 1 - (1 - unspread_bkg)**neighborhood_area
            eps = self.settings['min_probability']
            spread_back = np.clip(spread_back, eps, 1 - eps)

            weights = np.log(sub_kernels[0]/(1-sub_kernels[0]) * ((1-spread_back)/spread_back))
            weights_plus = np.clip(np.log(sub_kernels[0]/(1-sub_kernels[0]) * ((1-spread_back)/spread_back)), 0, np.inf)


            llhs = (feats0 * weights + feats0 * weights_plus * 4).sum()

        #means = feats0.reshape((-1, self.num_features)).mean(axis=0)
        #print lhsss  
        #print llhs, feats0.mean()
        if feats0.mean() < 0.02:
            return 0 
        else:
            return llhs

    def detect(self, img, fileobj=None, mixcomps=None):
        bbs = self.detect_coarse(img, fileobj=fileobj, mixcomps=mixcomps) 

        # Now, run it again and refine these probabilities
        if 0:
            for bbobj in bbs:
                new_score = self.calc_score(img, bbobj, score=bbobj.confidence)
                #print 'Score: ', bbobj.confidence, ' -> ', new_score    
                bbobj.confidence = new_score

        #print '-----------'
        #for bbobj in bbs:
        #    print bbobj.correct, bbobj.score
    
        return bbs

    def detect_coarse(self, img, fileobj=None, mixcomps=None):
        if mixcomps is None:
            mixcomps = range(self.num_mixtures)

        # TODO: Temporary stuff
        if 0:
            bbs = []
            for mixcomp in mixcomps:
                bbs0, resmap, feats, img_resized = self.detect_coarse_single_factor(img, 1.0, mixcomp, img_id=fileobj.img_id)
                bbs += bbs0

            # Do NMS here
            final_bbs = self.nonmaximal_suppression(bbs)
            
            # Mark corrects here
            if fileobj is not None:
                self.label_corrects(final_bbs, fileobj)


            return final_bbs
        else:
            # TODO: This does not use a Guassian pyramid, so it
            # resizes everything from scratch, which is MUCH SLOWER

            min_size = self.settings['min_size'] 
            min_factor = min_size / max(self.orig_kernel_size)
            max_size = self.settings['max_size'] 
            max_factor = max_size / max(self.orig_kernel_size)

            num_levels = 2
            factors = []
            skips = 0
            eps = 1e-8
            for i in xrange(1000):
                factor = self.settings['scale_factor']**(i-1)
                if factor > max_factor+eps:
                    break
                if factor >= min_factor-eps:
                    factors.append(factor) 
                else:
                    skips += 1
            num_levels = len(factors) + skips

            bbs = []
            for i, factor in enumerate(factors):
                #print 'factor', factor
                for mixcomp in mixcomps:
                    bbs0, resmap, feats, img_resized = self.detect_coarse_single_factor(img, factor, mixcomp, img_id=fileobj.img_id)
                    bbs += bbs0
        
            # Do NMS here
            final_bbs = self.nonmaximal_suppression(bbs)

            # Mark corrects here
            if fileobj is not None:
                self.label_corrects(final_bbs, fileobj)


            return final_bbs

        # ********************** OLD STUFF *****************************

        # Build image pyramid
        min_size = self.settings['min_size'] 
        min_factor = min_size / max(self.orig_kernel_size)#self.unpooled_kernel_side

        max_size = self.settings['max_size'] 
        max_factor = max_size / max(self.orig_kernel_size)#self.unpooled_kernel_side

        num_levels = 2
        factors = []
        skips = 0
        eps = 1e-8
        for i in xrange(1000):
            factor = self.settings['scale_factor']**i
            if factor > max_factor+eps:
                break
            if factor >= min_factor-eps:
                factors.append(factor) 
            else:
                skips += 1
        num_levels = len(factors) + skips

        ag.set_verbose(False)
        ag.info("Setting up pyramid")
        from skimage.transform import pyramid_gaussian 
        pyramid = list(pyramid_gaussian(img, max_layer=num_levels, downscale=self.settings['scale_factor']))[skips:]

        # Filter out levels that are below minimum scale

        # Prepare each level 
        def extract2(image):
            return self.descriptor.extract_features(image, dict(spread_radii=(0, 0), preserve_size=False))
        def extract(image):
            return self.descriptor.extract_features(image, dict(spread_radii=self.settings['spread_radii'], preserve_size=True))

        edge_pyramid = map(self.extract_spread_features, pyramid)
        ag.info("Getting edge pyramid")
        unspread_edge_pyramid = map(extract, pyramid)
        spread_edge_pyramid = map(extract, pyramid)

        ag.info("Extract background model")
        unspread_bkg_pyramid = map(self.bkg_model, unspread_edge_pyramid)
        spread_bkg_pyramid = map(lambda p: self.bkg_model(p, spread=True), spread_edge_pyramid)

        ag.info("Subsample")
        #small_pyramid = map(self.subsample, edge_pyramid) 

        bbs = []
        for i, factor in enumerate(factors):
            # Prepare the kernel for this mixture component
            ag.info("Prepare kernel", i, "factor", factor)
            sub_kernels = self.prepare_kernels(unspread_bkg_pyramid[i][0])

            for mixcomp in mixcomps:
                ag.info("Detect for mixture component", mixcomp)
            #for mixcomp in [1]:
                bbsthis, _ = self.detect_coarse_at_factor(edge_pyramid[i], sub_kernels, spread_bkg_pyramid[i][1], factor, mixcomp)
                bbs += bbsthis

        ag.info("Maximal suppression")
        # Do NMS here
        final_bbs = self.nonmaximal_suppression(bbs)
        
        # Mark corrects here
        if fileobj is not None:
            self.label_corrects(final_bbs, fileobj)


        return final_bbs

    def detect_coarse_at_factor(self, sub_feats, sub_kernels, spread_bkg, factor, mixcomp):
        # Get background level
        resmap = self.response_map(sub_feats, sub_kernels, spread_bkg, mixcomp, level=-1)

        kern = sub_kernels[mixcomp]

        # TODO: Decide this in a way common to response_map
        sh = kern.shape
        padding = (sh[0]//2, sh[1]//2, 0)

        # Get size of original kernel (but downsampled)
        full_sh = self.kernel_sizes[mixcomp]
        psize = self.settings['subsample_size']
        sh2 = sh
        sh = (full_sh[0]//psize[0], full_sh[1]//psize[1])

        th = -np.inf
        top_th = 200.0
        bbs = []

        agg_factors = tuple([psize[i] * factor for i in xrange(2)])
        bb_bigger = (0.0, 0.0, sub_feats.shape[0] * agg_factors[0], sub_feats.shape[1] * agg_factors[1])
        for i in xrange(resmap.shape[0]):
            for j in xrange(resmap.shape[1]):
                score = resmap[i,j]
                if score >= th:
                    i_corner = i-sh[0]//2
                    j_corner = j-sh[1]//2

                    index_pos = (i-padding[0], j-padding[1])

                    obj_bb = self.boundingboxes[mixcomp]
                    bb = [(i_corner + obj_bb[0]) * agg_factors[0],
                          (j_corner + obj_bb[1]) * agg_factors[1],
                          (i_corner + obj_bb[2]) * agg_factors[0],
                          (j_corner + obj_bb[3]) * agg_factors[1],
                    ]

                    # Clip to bb_bigger 
                    bb = gv.bb.intersection(bb, bb_bigger)
    
                    #score0 = score1 = 0
                    score0 = i
                    score1 = j

                    conf = score
                    dbb = gv.bb.DetectionBB(score=score, score0=score0, score1=score1, box=bb, index_pos=index_pos, confidence=conf, scale=factor, mixcomp=mixcomp)

                    if gv.bb.area(bb) > 0:
                        bbs.append(dbb)

        # Let's limit to five per level
        bbs_sorted = self.nonmaximal_suppression(bbs)
        bbs_sorted = bbs_sorted[:15]

        return bbs_sorted, resmap

    def response_map(self, sub_feats, sub_kernels, spread_bkg, mixcomp, level=0):
        kern = sub_kernels[mixcomp]
        if self.settings.get('per_mixcomp_bkg'):
            spread_bkg =  spread_bkg[mixcomp]

        sh = kern.shape
        padding = (sh[0]//2, sh[1]//2, 0)
        bigger = ag.util.zeropad(sub_feats, padding)

        # Since the kernel is spread, we need to convert the background
        # model to spread
        radii = self.settings['spread_radii']
        neighborhood_area = ((2*radii[0]+1)*(2*radii[1]+1))

        eps = self.settings['min_probability']
        #spread_bkg = np.clip(spread_bkg, eps, 1 - eps)

        
        # TEMP
        spread_bkg = np.clip(spread_bkg, eps, 1 - eps)
        kern = np.clip(kern, eps, 1 - eps) 
        #spread_bkg = np.load('new-bkg.npy')

        #spread_bkg *= 1.8
        if 0:
            spread_bkg = np.clip(spread_bkg, 0.02, 0.98)
            kern = np.clip(kern, 0.02, 0.98)

        #spread_bkg[:] = 0.025

        weights = np.log(kern/(1-kern) * ((1-spread_bkg)/spread_bkg))

        # Some experiments (will be removed)

        #weights -= weights.mean() * 1.25

        if 0:
            # - 
            wp = np.maximum(weights, 0)
            wm = np.minimum(weights, 0)

            wp /= wp.sum()
            wm /= np.fabs(wm.sum())

            weights = wp + 2 * wm
    
        #print 'mixcomp', mixcomp
        from .fast import multifeature_correlate2d
        #print bigger.shape, weights.shape
        #index = 26 
        #res = multifeature_correlate2d(bigger[...,index:index+1], weights[...,index:index+1].astype(np.float64)) 
        res = multifeature_correlate2d(bigger, weights.astype(np.float64))

        # Standardization
        testing_type = self.settings.get('testing_type', 'object-model')

        if testing_type == 'fixed':
            res -= self.fixed_train_mean[mixcomp]
            res /= self.fixed_train_std[mixcomp]
        elif testing_type == 'object-model':
            a = weights
            res -= (kern * a).sum()
            res /= np.sqrt((a**2 * kern * (1 - kern)).sum())
        elif testing_type == 'background-model':
            a = weights
            res -= (spread_bkg * a).sum()
            res /= np.sqrt((a**2 * spread_bkg * (1 - spread_bkg)).sum())
        elif testing_type == 'zero-model':
            pass
        elif testing_type == 'none':
            # We need to add the constant term that isn't included in weights
            res += np.log((1 - kern) / (1 - spread_bkg)).sum() 

        return res

    def nonmaximal_suppression(self, bbs):
        # This one will respect scales a bit more
        bbs_sorted = sorted(bbs, reverse=True)

        overlap_threshold = self.settings.get('overlap_threshold', 0.5)
        #print "TEMP TEMP TEMP TEMP!!!"

        # Suppress within a radius of H neighboring scale factors
        sf = self.settings['scale_factor']
        H = self.settings.get('scale_suppress_radius', 1)
        i = 1
        lo, hi = 1/(H*sf)-0.01, H*sf+0.01
        while i < len(bbs_sorted):
            # TODO: This can be vastly improved performance-wise
            area_i = gv.bb.area(bbs_sorted[i].box)
            for j in xrange(i):
                # VERY TEMPORARY: This avoids suppression between classes
                #if bbs_sorted[i].mixcomp != bbs_sorted[j].mixcomp:
                #    continue
        
                overlap = gv.bb.area(gv.bb.intersection(bbs_sorted[i].box, bbs_sorted[j].box))/area_i
                scale_diff = (bbs_sorted[i].scale / bbs_sorted[j].scale)
                if overlap > overlap_threshold and lo <= scale_diff <= hi: 
                    del bbs_sorted[i]
                    i -= 1
                    break

            i += 1
        return bbs_sorted

    def bounding_box_for_mix_comp(self, k):
        """This returns a bounding box of the support for a given component"""

        # Take the bounding box of the support, with a certain threshold.
        #print("Using alpha", self.use_alpha, "support", self.support)
        if self.support is not None:
            supp = self.support[k] 
            supp_axs = [supp.max(axis=1-i) for i in xrange(2)]

            th = self.settings['bounding_box_opacity_threshold']
            # Check first and last value of that threshold
            bb = [np.where(supp_axs[i] > th)[0][[0,-1]] for i in xrange(2)]

            # This bb looks like [(x0, x1), (y0, y1)], when we want it as (x0, y0, x1, y1)
            psize = self.settings['subsample_size']
            ret = (bb[0][0]/psize[0], bb[1][0]/psize[1], bb[0][1]/psize[0], bb[1][1]/psize[1])
            return ret
        else:
            psize = self.settings['subsample_size']
            return (0, 0, self.orig_kernel_size[0]/psize[0], self.orig_kernel_size[1]/psize[1])

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
                            best_bbobj = bb1obj 
                            best_bb = bb1

            if best_bbobj is not None:
                bb2obj.correct = True
                bb2obj.difficult = best_bbobj.difficult
                # Don't count difficult
                if not best_bbobj.difficult:
                    tot += 1
                used_bb.add(best_bb)

    def _preprocess(self):
        """Pre-processes things"""
        # Prepare bounding boxes for all mixture model
        self.boundingboxes = np.array([self.bounding_box_for_mix_comp(i) for i in xrange(self.num_mixtures)])

    @classmethod
    def load_from_dict(cls, d):
        try:
            num_mixtures = d['num_mixtures']
            descriptor_cls = gv.BinaryDescriptor.getclass(d['descriptor_name'])
            if descriptor_cls is None:
                raise Exception("The descriptor class {0} is not registered".format(d['descriptor_name'])) 
            descriptor = descriptor_cls.load_from_dict(d['descriptor'])
            obj = cls(num_mixtures, descriptor)
            mix_dict = d.get('mixture')
            if mix_dict is not None:
                obj.mixture = ag.stats.BernoulliMixture.load_from_dict(d['mixture'])
            else:
                obj.mixture = None
            obj.settings = d['settings']
            obj.orig_kernel_size = d.get('orig_kernel_size')
            obj.kernel_basis = d.get('kernel_basis')
            obj.kernel_basis_samples = d.get('kernel_basis_samples')
            obj.kernel_templates = d.get('kernel_templates')
            obj.kernel_sizes = d.get('kernel_sizes')
            obj.use_alpha = d['use_alpha']
            obj.support = d.get('support')

            obj.fixed_bkg = d.get('fixed_bkg')
            obj.fixed_spread_bkg = d.get('fixed_spread_bkg')

            obj.fixed_train_std = d.get('fixed_train_std')
            obj.fixed_train_mean = d.get('fixed_train_mean')

            obj._preprocess()
            return obj
        except KeyError as e:
            # TODO: Create a new exception for these kinds of problems
            raise Exception("Could not reconstruct class from dictionary. Missing '{0}'".format(e))

    def save_to_dict(self):
        d = {}
        d['num_mixtures'] = self.num_mixtures
        d['descriptor_name'] = self.descriptor.name
        d['descriptor'] = self.descriptor.save_to_dict()
        if self.mixture is not None:
            d['mixture'] = self.mixture.save_to_dict(save_affinities=True)
        d['orig_kernel_size'] = self.orig_kernel_size
        d['kernel_templates'] = self.kernel_templates
        d['kernel_basis'] = self.kernel_basis
        d['kernel_basis_samples'] = self.kernel_basis_samples
        d['kernel_sizes'] = self.kernel_sizes
        d['use_alpha'] = self.use_alpha
        d['support'] = self.support
        d['settings'] = self.settings

        if self.fixed_bkg is not None:
            d['fixed_bkg'] = self.fixed_bkg

        if self.fixed_spread_bkg is not None:
            d['fixed_spread_bkg'] = self.fixed_spread_bkg 

        if self.settings['testing_type'] == 'fixed':
            d['fixed_train_std'] = self.fixed_train_std
            d['fixed_train_mean'] = self.fixed_train_mean

        return d
