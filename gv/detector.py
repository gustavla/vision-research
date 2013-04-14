from __future__ import division
from __future__ import print_function
from __future__ import absolute_import 

# TODO: Temporary
#import matplotlib
#matplotlib.use("Agg")
import matplotlib.pylab as plt

import amitgroup as ag
import numpy as np
import scipy.signal
from .saveable import Saveable
import gv
import sys
from copy import deepcopy

def _along_kernel(direction, radius):
    d = direction%4
    kern = None
    if d == 2: # S/N
        kern = np.zeros((radius*2+1,)*2, dtype=np.uint8)
        kern[radius,:] = 1
    elif d == 0: # E/W
        kern = np.zeros((radius*2+1,)*2, dtype=np.uint8)
        kern[:,radius] = 1
    elif d == 3: # SE/NW
        kern = np.eye(radius*2+1, dtype=np.uint8)[::-1]
    elif d == 1: # NE/SW
        kern = np.eye(radius*2+1, dtype=np.uint8)
            
    return kern

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

def _integrate(ii, r0, c0, r1, c1):
    """Use an integral image to integrate over a given window.

    Parameters
    ----------
    ii : ndarray
    Integral image.
    r0, c0 : int
    Top-left corner of block to be summed.
    r1, c1 : int
    Bottom-right corner of block to be summed.

    Returns
    -------
    S : int
    Integral (sum) over the given window.

    """
    # This line is modified
    S = np.zeros(ii.shape[-1]) 

    S += ii[r1, c1]

    if (r0 - 1 >= 0) and (c0 - 1 >= 0):
        S += ii[r0 - 1, c0 - 1]

    if (r0 - 1 >= 0):
        S -= ii[r0 - 1, c1]

    if (c0 - 1 >= 0):
        S -= ii[r1, c0 - 1]

    return S

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

#TODO :Temporary, remove
#SUPP = np.load('fa1ming2-model02.npy').flat[0]['support']

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
        return self.kernel_templates.shape[-1]

    def load_img(self, images, offsets=None):
        resize_to = self.settings.get('image_size')
        for i, img_obj in enumerate(images):
            if isinstance(img_obj, str):
                print("Image file name", img_obj)
                img = gv.img.load_image(img_obj)
            grayscale_img = gv.img.asgray(img)

            # TODO: Experimental
            # Blur the grayscale_img
            #grayscale_img = ag.util.blur_image(grayscale_img, 0.05)

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
            #, alpha

            yield i, grayscale_img, img, alpha

    def gen_img(self, images, actual=False, use_mask=False): #TODO: temp2
        if use_mask:
            for i, grayscale_img, img, alpha in self.load_img(images):
                #ag.info("Mixing image {0}".format(i))
                #alpha = (img[...,3] > 0.2).astype(np.uint8)

                # Dilate alpha here
                #alpha = ag.util.inflate2d(alpha, np.ones((5, 5))) 

                #alpha = None
                final_feats = self.extract_unspread_features(grayscale_img, support_mask=alpha)
                #if final_edges[-1].mean() > 0.5:
                #    import pdb; pdb.set_trace()

                yield final_feats
        elif 0:
            for i, grayscale_img, img, alpha in self.load_img(images):

                # First, extract unspread edges
                bsett = self.descriptor.bedges_settings()
                sett = bsett.copy()

# self.settings['parts']['edges'].copy()
                sett['radius'] = 0
                front_edges = ag.features.bedges(grayscale_img, **sett) 

                # Now, inject more edges
                bkg_edges = (np.random.random(front_edges.shape) < 0.35)
            
                alpha = (img[...,3] > 0.1)
                alpha = alpha[2:-2,2:-2]
                alpha = alpha.reshape(alpha.shape + (1,))
        
                edges = alpha * front_edges + (1 - alpha) * bkg_edges
    
                # Now do the spreading


                # Dump to file
                plt.clf()
                ag.plot.images(np.rollaxis(edges, axis=2)) 
                plt.savefig('dumpy/img-{0}.png'.format(i))

                # Propagate the feature along the edge 
                for j in xrange(bkg_edges.shape[-1]):
                    kernel = _along_kernel(j, bsett['radius'])
                    edges[...,j] = ag.util.inflate2d(edges[...,j], kernel)
                
                feats = self.descriptor.extract_parts(edges.astype(np.uint8), settings={'spread_radii': (0, 0)})

                yield feats 
        else:
            for i, grayscale_img, img, alpha in self.load_img(images):
                #ag.info("Mixing image {0}".format(i))
                if self.train_unspread and not actual:
                    final_edges = self.extract_unspread_features(grayscale_img)
                else:
                    final_edges = self.extract_spread_features(grayscale_img)
                    if actual:
                        final_edges = self.subsample(final_edges)
                yield final_edges

    def train_from_images(self, images):
        self.orig_kernel_size = None

        mixture, kernel_templates, support = self._train(images)

        if self.settings.get('recenter'):
            #mixture.templates = np.empty(0)
            radii = (2, 2)
            psize = (2, 2)

            fix_bkg = self.settings.get('fixed_background_probability')
            bkg = 1 - (1 - fix_bkg)**((2 * radii[0] + 1) * (2 * radii[1] + 1))

            search_size = 10 

            # TODO: This is a hack. Refactor.
            self.kernel_templates = kernel_templates
            self.support = support

            kernels = self.prepare_kernels(np.ones(kernel_templates.shape[-1])*bkg, settings=dict(spread_radii=radii, subsample_size=psize))

            offsets = np.zeros((len(images), 2), dtype=np.int32)

            for i, grayscale_img, img, alpha in self.load_img(images):
                k = mixture.which_component(i)

                # Make abstraction
                orig_feats = self.extract_spread_features(grayscale_img, dict(spread_radii=radii))
                feats = gv.sub.subsample(orig_feats, psize)

                #print(feats.shape)

                padded = ag.util.zeropad(feats, (search_size, search_size, 0))
                 
                # Check log-likelihood of that image with different shifts
                from .fast import multifeature_correlate2d 
                kernel = mixture.templates[k].reshape(feats.shape[:2] + (-1,)).astype(np.float64)
                #res = multifeature_correlate2d(padded, np.log(kernel))
                res = multifeature_correlate2d(padded, np.log(kernel/(1-kernel) * ((1-bkg)/bkg)))
                #res += multifeature_correlate2d(1-padded, np.log(1-kernel))
 
                # Get max
                best = np.unravel_index(np.argmax(res), res.shape)
                offset = (best[0]-search_size, best[1]-search_size)
                offsets[i] = offset
                print(i, 'offset', offset)

                #import pudb; pudb.set_trace()
            
                #print(res)

            # Re-train centered        
            mixture, kernel_templates, support = self._train(images, offsets)

        self.mixture = mixture
        self.kernel_templates = kernel_templates
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
                #self.use_alpha = False # TODO: Temp2
                if self.use_alpha:
                    alpha_maps = np.empty((len(images),) + img.shape[:2], dtype=np.uint8)

            if self.use_alpha:
                a = (img[...,3] > 0.05).astype(np.uint8)

                # Erode the alpha map
                #import skimage.morphology
                #a = skimage.morphology.erosion(a, np.ones((9, 9), dtype=np.uint8))

                alpha_maps[i] = a

            if self.train_unspread:
                final_edges = self.extract_unspread_features(grayscale_img)
                # TODO: Temporary stuff
                #alpha_edges = self.extract_unspread_features(img[...,3])
                #invalpha_edges = self.extract_unspread_features(1-img[...,3])
            
                #final_edges |= alpha_edges | invalpha_edges
            else:
                final_edges = self.extract_spread_features(grayscale_img)

            #else:
                #final_edges = (np.random.random(651264 * 4)>0.5).astype(np.uint8)
            #edges = self.subsample(self.extract_spread_features(grayscale_img))
            #import scipy.sparse
            orig_edges = self.extract_spread_features(grayscale_img)
            edges = gv.sub.subsample(orig_edges, (2, 2)).ravel()

            #real_shape = orig_edges.shape
            #real_shape = orig_edges.shape
            #edges = self.descriptor.extract_features(grayscale_img)

            if self.orig_kernel_size is None:
                #self.orig_kernel_size = (img.shape[0]//psize[0], img.shape[1]//psize[1]) #orig_edges.shape[:2]
                self.orig_kernel_size = (img.shape[0], img.shape[1]) #orig_edges.shape[:2]
        
            # Extract the parts, with some pooling 
            #small = self.descriptor.pool_features(edges)
            if shape is None:
                shape = edges.shape
                if sparse:
                    if build_sparse:
                        output = scipy.sparse.dok_matrix((len(images),) + edges.shape, dtype=np.uint8)
                    else:
                        output = np.zeros((len(images),) + edges.shape, dtype=np.uint8)
                else:
                    output = np.empty((len(images),) + edges.shape, dtype=np.uint8)

                orig_output = np.empty((len(images),) + orig_edges.shape, dtype=np.uint8)
            
            orig_output[i] = orig_edges
                
            #assert edges.shape == shape, "Images must all be of the same size, for now at least"
            
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

        # Train mixture model OR SVM
        mixture = ag.stats.BernoulliMixture(self.num_mixtures, output, float_type=np.float32)

        minp = 1e-5
        # TODO: This should check for spreading correction, which can be done without use of alpha
        #if not self.use_alpha:
        #   minp = self.settings['min_probability']
        
        mixture.run_EM(1e-8, minp)

        #mixture.templates = np.empty(0)

        # Now create our unspread kernels
        # Remix it - this iterable will produce each object and then throw it away,
        # so that we can remix without having to ever keep all mixing data in memory at once
             
        use_mask = False 
        kernel_templates = np.clip(mixture.remix_iterable(self.gen_img(images, use_mask=use_mask)), 1e-5, 1-1e-5) # TDOO: Temp2

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
        fix_bkg = self.settings.get('fixed_background_probability')
        radii = self.settings['spread_radii']
        if fix_bkg is not None:
            #flat = output.reshape((output.shape[0], -1))
            #flat_template = mixture.kernel_templates.reshape((mixture.num_mixtures, -1)) 

            if 0:
                bkg_llh = output.sum(axis=1) * np.log(bkg) + (1-output).sum(axis=1) * np.log(1-bkg)
                 
                lrt = mixture.mle - bkg_llh

                self.train_std = lrt.std()
                self.train_mean = lrt.mean()

                llhs = np.zeros((mixture.num_mix, output.shape[0]))
                kernel_templates.shape[0]
                for k in xrange(mixture.num_mix):
                    for i in xrange(output.shape[0]):
                        kern = kernel_templates[k].ravel()
                        X = orig_output[i].ravel()
                        llhs[k,i] = np.sum( X * np.log(kern/bkg) + (1-X) * np.log((1-kern)/(1-bkg)) )


            L = len(self.settings['levels'])
            self.train_mean = np.zeros(L)
            self.train_std = np.zeros(L)

            for i, (sub, spread) in enumerate(self.settings['levels']):

                psize = (sub,)*2
                radii = (spread,)*2

                self.settings['subsample_size'] = psize
                self.settings['spread_radii'] = radii 

                orig_output = None
                # Get images with the right amount of spreading
                for j, grayscale_img, img, alpha in self.load_img(images, offsets):
                    orig_edges = self.extract_spread_features(grayscale_img, settings=dict(spread_radii=radii))
                    if orig_output is None:
                        orig_output = np.empty((len(images),) + orig_edges.shape, dtype=np.uint8)
                    orig_output[j] = orig_edges
                        #orig_edges = self.extract_spread_features(grayscale_img)
                    

                bkg = 1 - (1 - fix_bkg)**((2 * radii[0] + 1) * (2 * radii[1] + 1))
                #bkg = 0.05

                self.kernel_templates = kernel_templates
                kernels = self.prepare_kernels(np.ones(kernel_templates.shape[-1])*bkg, settings=dict(spread_radii=radii, subsample_size=psize))

                sub_output = gv.sub.subsample(orig_output, psize, skip_first_axis=True)

                #import pylab as plt
                #plt.imshow(kernels[0].sum(axis=-1), interpolation='nearest')
                #plt.show()

                #print('sub_output', sub_output.shape)
                theta = kernels.reshape((kernels.shape[0], -1))
                X = sub_output.reshape((sub_output.shape[0], -1))

                try:
                    llhs = np.dot(X, np.log(theta/(1-theta) * ((1-bkg)/bkg)).T)
                except:
                    import pdb; pdb.set_trace()
                #C = np.log((1-theta)/(1-bkg)).sum(axis=1)
                #llhs += C
                
                lrt = llhs.max(axis=1)

                #print(i, 'pixels', X.shape[1])
                #print('min_lrt', lrt.min())

                self.train_mean[i] = lrt.mean()
                self.train_std[i] = lrt.std()

                #import pylab as plt
                #plt.hist(lrt)
                #plt.show()

            #import pdb; pdb.set_trace()
#
            print("mean", self.train_mean)
            print("std", self.train_std)
            #print()
            #print("mean2", stuff.mean())
            #print("std2", stuff.std()) 


            #self.train_std = lrt.std()
            #self.train_mean = lrt.mean()

            #for i in xrange(self.output.shape[0]):
                #edges = self.output.shape[i]
                
        
    
        return mixture, kernel_templates, support

    def _preprocess_pooled_support(self):
        """
        Pools the support to the same size as other types of pooling. However, we use
        mean pooling here, which might be different from the pooling we use for the
        features.
        """

        if self.use_alpha:
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

    def extract_unspread_features(self, image, support_mask=None):
        edges = self.descriptor.extract_features(image, {'spread_radii': (0, 0)}, support_mask=support_mask)
        return edges

    def extract_spread_features(self, image, settings={}):

        #self.back = image.sum(axis=0).sum(axis=0) / np.prod(image.shape[:2])

        sett = self.settings.copy()
        sett.update(settings)
    
        if 0:
            th = 0
            N = 100
            p = 0.05
            
            for n in xrange(N):
                p2 = 1 - np.sum([scipy.misc.comb(N, i) * p**i * (1-p)**(N-i) for i in xrange(n)])
                if p2 < 0.01:
                    th = n
                    break

            print('Threshold', th)

        edges = self.descriptor.extract_features(image, {'spread_radii': sett['spread_radii']})
        return edges 

    @property
    def unpooled_kernel_size(self):
        return (self.kernel_templates.shape[1], self.kernel_templates.shape[2])

    @property
    def unpooled_kernel_side(self):
        return max(self.unpooled_kernel_size)


    def background_model(self, edges, settings={}):
        num_features = edges.shape[-1]
        
        sett = self.settings.copy()  
        sett.update(settings)

        fix_bkg =  sett.get('fixed_background_probability')
        #print("fix_bkg", fix_bkg)
        radii = sett['spread_radii']
        #print("radii", radii)
        back = np.empty(num_features)
        for f in xrange(num_features):
            back[f] = edges[...,f].sum()
        back /= np.prod(edges.shape[:2])

        if fix_bkg is not None:
            test_back = 1 - (1 - fix_bkg)**((2 * radii[0] + 1) * (2 * radii[1] + 1))
            back[:] = test_back

        else:
            test_back = back
            #for f in xrange(num_features):
                #back[f] = edges[...,f].sum()
            #back /= np.prod(edges.shape[:2])

        if 0:
            #import pylab as plt
            K = 2 
            flat_edges = edges.reshape((np.prod(edges.shape[:2]),-1))
            backmodel = ag.stats.BernoulliMixture(K, flat_edges)
            backmodel.run_EM(1e-4, 0.05)


            #aa = np.argmax(backmodel.affinities.reshape(edges.shape[:2]+(-1,)), axis=-1)
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
            back = backmodel.templates[back_i]

        eps = sett['min_probability']
        # Do not clip it here.
        back = np.clip(back, eps, 1-eps)
        test_back = np.clip(test_back, eps, 1-eps)
        #back[:] = 0.05
    
        return back, test_back

    def subsample(self, edges):
        return gv.sub.subsample(edges, self.settings['subsample_size'])

    def prepare_kernels(self, back, settings={}):
        #print('..............',back)
        num_features = self.descriptor.num_features
        sett = self.settings.copy()
        sett.update(settings) 

        # TODO: Very temporary
        #back = np.load('bkg.npy')

        if self.use_basis:
            pass
        else:
            kernels = self.kernel_templates.copy()

        eps = sett['min_probability']
        psize = sett['subsample_size']

        def weighted_choice(x):
            xcum = np.cumsum(x) 
            r = np.random.uniform(0, xcum[-1])
            return np.where(r < xcum)[0][0]

        def weighted_choice_unit(x):
            xcum = np.cumsum(x) 
            r = np.random.random()
            w = np.where(r < xcum)[0]
            if w.size == 0:
                return -1
            else:
                return w[0]
            return 

        part_size = self.descriptor.settings['part_size']

        # First, translate kernel
        DO_NEW = self.settings.get('do_new', False)# TODO: New stuff
        if DO_NEW: # Translate kernel to incorporate background model
            do_it = False
            try:
                kernels = np.load('kernel_cache.npy')
            except IOError: 
                do_it = True 

            if do_it:
                np.save('kernel_before.npy', kernels)

                offsets = gv.sub.subsample_offset(kernels[0], psize)#[psize[i]//2 for i in xrange(2)]
                sh = kernels.shape[1:3]
                # Number of samples
                N = 1000
                part_counts = np.zeros(num_features)
                num_edges = self.descriptor.parts.shape[-1]

                old_kernels = kernels

                if 0:
                    good_back_spread = np.load('bkg.npy')
                    radii = sett['spread_radii']
                    neighborhood_area = ((2*radii[0]+1)*(2*radii[1]+1))
                    #back = np.load('bkg.npy')
                    good_back = 1 - (1 - good_back_spread)**(1/neighborhood_area)
                else:
                    good_back = np.load('bkg2_nospread.npy')


                def get_probs(f):
                    if f == -1:
                        return 0
                    else:
                        return self.descriptor.parts[f]
                
                print("Doing one") 
                for mixcomp in xrange(self.num_mixtures):
                    # Note, we are going in strides of psize, given a a certain offset, since
                    # we will be subsampling anyway, so we don't need to do the rest.
                    for i in xrange(sh[0]):# (offsets[0], sh[0], psize[0]):
                        for j in xrange(sh[1]):#(offsets[1], sh[1], psize[1]):
                            print("Doing pixel {0}/{1}, {2}/{3}".format(i, sh[0], j, sh[1]))
                            # TODO: Part_size is hard-coded. Clean up.
                            p_alpha = self.full_support[mixcomp, 6+i-4:6+i+5, 6+j-4:6+j+5]
                            p_kernel = kernels[mixcomp,i,j]
                            alpha_ij = self.full_support[mixcomp, 6+i, 6+j]
                            p_back = good_back 

                            if True: #0 < p_alpha.mean() < 0.99:
                                part_counts = np.zeros(num_features)
                                for loop in xrange(N):
                                    #import pudb; pudb.set_trace()
                                    f_obj = weighted_choice_unit(p_kernel)
                                    probs_obj = get_probs(f_obj)
                                    if alpha_ij > 0:
                                        probs_obj /= alpha_ij 

                                    f_bkg = weighted_choice_unit(good_back)
                                    probs_bkg = get_probs(f_bkg) 

                                    #import pudb; pudb.set_trace()
                                    
                                    if f_obj == -1 and f_bkg == -1:
                                        pass # Add nothing
                                    elif f_obj == -1 and f_bkg != -1:
                                        part_counts[f_bkg] += 1
                                    elif f_obj != -1 and f_bkg == -1:
                                        part_counts[f_obj] += 1 
                                    else:
                                        # Draw from the alpha
                                        A = (np.random.random(p_alpha.shape) < p_alpha).astype(np.uint8) 
                                        AA = A.reshape(A.shape + (1,))

                                        probs_mixed = AA * probs_obj + (1 - AA) * probs_bkg 

                                        # Draw samples from the mixture components
                                        X = (np.random.random(probs_mixed.shape) < probs_mixed).astype(np.uint8)
                    
                                        # Check which part this is most similar to
                                        scores = np.apply_over_axes(np.sum, X * np.log(self.descriptor.parts) + (1 - X) * np.log(1 - self.descriptor.parts), [1, 2, 3]).ravel()
                                        f_best = np.argmax(scores)
                                        #f_best = np.argmax(np.apply_over_axes(np.sum, np.fabs(self.descriptor.parts - X), [1, 2, 3]).ravel())
                                        part_counts[f_best] += 1
                                        
                                        #p = _integrate(integral_aa_log[mixcomp], i, j, i+istep, j+jstep)

                                kernels[mixcomp,i,j] = part_counts / N

                print("Done")
            
                np.save('kernel_cache.npy', kernels)
                 

        #if not self.use_alpha:
            # Do not correct for background
            #pass
        #elif self.train_unspread:
        if self.train_unspread:
            radii = sett['spread_radii']
            neighborhood_area = ((2*radii[0]+1)*(2*radii[1]+1))
            #back = np.load('bkg.npy')
            nospread_back = 1 - (1 - back)**(1/neighborhood_area)
            
            # TODO: Use this background instead.
            #nospread_back = np.load('bkg.npy')

            if self.use_basis:
                C = self.kernel_basis * np.expand_dims(nospread_back, -1)
                kernels = C.sum(axis=-2) / self.kernel_basis_samples

            # Clip nospread_back, since we didn't clip it before
            #nospread_back = np.clip(nospread_back, eps, 1-eps)

            #print('self.support.shape', self.support.shape) 
            #print('kernels', kernels.shape) 

            if self.use_alpha:
                invsupp = (1-self.support.reshape(self.support.shape+(1,)))
            else:
                invsupp = 0

            if DO_NEW:
                invsupp = 0

            #aa_log = np.log((1 - kernels) - nospread_back * supp)
            aa_log = np.log(np.clip((1 - kernels) - nospread_back * invsupp, 0.00001, 1-0.00001))
            aa_log = ag.util.multipad(aa_log, (0, radii[0], radii[1], 0), np.log(1-nospread_back))
            integral_aa_log = aa_log.cumsum(1).cumsum(2)

            offsets = gv.sub.subsample_offset(kernels[0], psize)#[psize[i]//2 for i in xrange(2)]

            # Fix kernels
            istep = 2*radii[0]
            jstep = 2*radii[1]
            sh = kernels.shape[1:3]
            for mixcomp in xrange(self.num_mixtures):
                # Note, we are going in strides of psize, given a certain offset, since
                # we will be subsampling anyway, so we don't need to do the rest.
                for i in xrange(offsets[0], sh[0], psize[0]):
                    for j in xrange(offsets[1], sh[1], psize[1]):
                        p = _integrate(integral_aa_log[mixcomp], i, j, i+istep, j+jstep)
                        kernels[mixcomp,i,j] = 1 - np.exp(p)

            # TODO: Temporary stuff
            #import pdb; pdb.set_trace()
            #s = (SUPP[0] > 0).reshape(SUPP[0].shape + (1,))

            #s = SUPP[0].reshape(SUPP[0].shape + (1,))
            #kernels[0] = kernels[0] * s + back * (1 - s)

        else:
            # This does not handle support correctly when spreading
            for mixcomp in xrange(self.num_mixtures):
                if self.support is not None:
                    for f in xrange(num_features):
                        # This is explained in writeups/cad-support/.
                        kernels[mixcomp,...,f] += (1-self.support[mixcomp]) * back[f]
                        #kernels[mixcomp,...,f] = 1 - (1 - kernels[mixcomp,...,f]) * (1 - back[f])**(1-self.support[mixcomp])
            

        # Subsample kernels
        sub_kernels = gv.sub.subsample(kernels, psize, skip_first_axis=True)
        #import pdb; pdb.set_trace()


        sub_kernels = np.clip(sub_kernels, eps, 1-eps)

        # TEMP: Prepare means
        #self.means = np.array([ 35.93866455,  38.39765241,  39.96420018,  25.06870656,  32.96915432]) / 30.0
        #self.means = np.apply_over_axes(np.sum, self.support, [1, 2]).ravel()
        #self.means /= self.means.mean()
        K = self.settings.get('quantize_bins')
        if K is not None:
            sub_kernels = np.round(1+sub_kernels*(K-2))/K


        return sub_kernels

    def detect_coarse_single_factor(self, img, factor, mixcomp, img_id=0):
        """
        TODO: Experimental changes under way!
        """

        from skimage.transform import pyramid_reduce
        if abs(factor-1) < 1e-8:
            img_resized = img
        else:
            img_resized = pyramid_reduce(img, downscale=factor)

        #sh = gv.sub.subsample_size(img, (toplevel[0],)*2)
        # TODO: This is hack to make sub_ok same size as resmap. Investigate closer instead!
        ok = np.ones(tuple(np.asarray(img.shape)+2*np.ones(2)))

        last_resmap = None

        sold = self.settings.copy()

        resmaps = []

        # Coarse to fine structure
        if self.settings.get('coarse_to_fine'):
            levels = self.settings['levels']
            #plt.clf()
            #plt.subplot(421)
            #plt.imshow(img, interpolation='nearest', cmap=plt.cm.gray)

            for i, (sub, spread) in enumerate(levels[:-1]):
                psize = (sub,)*2
                radii = (spread,)*2

                self.settings['subsample_size'] = psize
                self.settings['spread_radii'] = radii 

                def extract(image):
                    #return self.descriptor.extract_features(image, dict(spread_radii=self.settings['spread_radii'], preserve_size=True))
                    return self.descriptor.extract_features(image, dict(spread_radii=radii, preserve_size=True))

                up_feats = extract(img_resized)
                #print('up_feats', up_feats.shape) 
                bkg, test_bkg = self.background_model(up_feats, settings=dict(spread_radii=radii))
            
                #print('bkg', bkg[0])
                #feats = self.subsample(up_feats, ) 
                feats = gv.sub.subsample(up_feats, psize) 

                # Prepare kernel
                sub_kernels = self.prepare_kernels(bkg, settings=dict(spread_radii=radii, subsample_size=psize))
        
                #import pylab as plt
                #plt.imshow(sub_kernels[0].sum(axis=-1), interpolation='nearest')
                #plt.show()

                #print('feats', feats.shape) 

                #print('sub_kernels', sub_kernels.shape)

                ok_sub = gv.sub.subsample(ok, psize)

                resmap, resplus = self.response_map(feats, sub_kernels, test_bkg, mixcomp, level=i, ok=ok_sub)
                
                resmaps.append(resmap)

                #print('ok', ok.shape)
                #print('resmap', resmap.shape)
                #print('psize', psize)
                # Do not actually run CTF
                if 0:
                    if i != len(levels) - 1:
                        gv.sub.erase(ok, resmap > -5, psize)
                        if i == 0:
                            gv.sub.erase(ok, resplus > -5, psize)

                if 0:
                    #import matplotlib.pylab as plt
                    plt.subplot(423+i*2)
                    plt.imshow(resmap, interpolation='nearest')
                    plt.colorbar()
                    plt.subplot(423+i*2+1)
                    plt.imshow(ok.copy(), interpolation='nearest')
                    last_resmap = resmap
            #plt.show()
            #plt.savefig("plogs/{0:03d}-{1}.png".format(img_id, mixcomp)) 


            
        self.settings = sold
        psize = self.settings['subsample_size']
        radii = self.settings['spread_radii']

        def extract(image):
            #return self.descriptor.extract_features(image, dict(spread_radii=self.settings['spread_radii'], preserve_size=True))
            return self.descriptor.extract_features(image, dict(spread_radii=radii, preserve_size=True))
        # Last psize
        ok_sub = gv.sub.subsample(ok, psize)

        up_feats = extract(img_resized)
        #print('up_feats', up_feats.shape) 
        bkg, test_bkg = self.background_model(up_feats, settings=dict(spread_radii=radii))
        feats = gv.sub.subsample(up_feats, psize) 
        sub_kernels = self.prepare_kernels(bkg, settings=dict(spread_radii=radii, subsample_size=psize))

        #import sys
        #sys.exit(0)
        bbs, resmap = self.detect_coarse_at_factor(feats, sub_kernels, test_bkg, factor, mixcomp, ok=ok_sub, resmaps=resmaps)

        final_bbs = bbs
        #print(len(bbs), '->', len(final_bbs))
    
        #resmap = last_resmap
        
        #final_bbs = []
        

        # Do NMS here
        #final_bbs = self.nonmaximal_suppression(bbs)
        #final_bbs = bbs

        return final_bbs, resmap, feats, img_resized

    def detect_coarse(self, img, fileobj=None, mixcomps=None):
        if mixcomps is None:
            mixcomps = range(self.num_mixtures)

        # TODO: Temporary stuff
        if 1:
            bbs = []
            for mixcomp in mixcomps:
                bbs0, resmap, feats, img_resized = self.detect_coarse_single_factor(img, 1.0, mixcomp, img_id=fileobj.img_id)
                bbs += bbs0
            #bbs2, resmap, feats, img_resized = self.detect_coarse_single_factor(img, 1.0, 1, img_id=fileobj.img_id)

            # Do NMS here
            final_bbs = self.nonmaximal_suppression(bbs)
            
            # Mark corrects here
            if fileobj is not None:
                self.label_corrects(final_bbs, fileobj)


            return final_bbs
        

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
        def extract(image):
            return self.descriptor.extract_features(image, dict(spread_radii=self.settings['spread_radii'], preserve_size=True))

        #edge_pyramid = map(self.extract_spread_features, pyramid)
        ag.info("Getting edge pyramid")
        edge_pyramid = map(extract, pyramid)
        ag.info("Extract background model")
        bkg_pyramid = map(self.background_model, edge_pyramid)
        ag.info("Subsample")
        small_pyramid = map(self.subsample, edge_pyramid) 

        bbs = []
        for i, factor in enumerate(factors):
            # Prepare the kernel for this mixture component
            ag.info("Prepare kernel", i, "factor", factor)
            sub_kernels = self.prepare_kernels(bkg_pyramid[i][0])

            for mixcomp in mixcomps:
                ag.info("Detect for mixture component", mixcomp)
            #for mixcomp in [1]:
                bbsthis, _ = self.detect_coarse_at_factor(small_pyramid[i], sub_kernels, test_bkg_pyramid[i][1], factor, mixcomp)
                bbs += bbsthis

        ag.info("Maximal suppression")
        # Do NMS here
        final_bbs = self.nonmaximal_suppression(bbs)
        
        # Mark corrects here
        if fileobj is not None:
            self.label_corrects(final_bbs, fileobj)


        return final_bbs

    def detect_coarse_at_factor(self, sub_feats, sub_kernels, back, factor, mixcomp, ok=None, resmaps=None):
        # Get background level
        resmap, resplus = self.response_map(sub_feats, sub_kernels, back, mixcomp, level=-1, ok=ok)

        #print('resmap', resmap.shape)

        # TODO: Remove edges
        sh = sub_kernels.shape[1:3]
        #resmap2 = resmap.min()*np.ones(resmap.shape)
        #resmap2[sh[0]//2:-sh[0]//2, sh[1]//2:-sh[1]//2] = resmap[sh[0]//2:-sh[0]//2, sh[1]//2:-sh[1]//2]
        #resmap = resmap2

        #resmap /= self.means[mixcomp]

        #prin

        th = -np.inf
        #top_th = resmap.max()#200.0
        top_th = 200.0
        bbs = []

        #nn_resmaps = np.zeros((2,) + resmap.shape)
        #if resmaps is not None:
        #    nn_resmaps[0] = ag.util.nn_resample2d(resmaps[0], resmap.shape)
        #    nn_resmaps[1] = ag.util.nn_resample2d(resmaps[1], resmap.shape)
        
        psize = self.settings['subsample_size']
        agg_factors = tuple([psize[i] * factor for i in xrange(2)])
        bb_bigger = (0.0, 0.0, sub_feats.shape[0] * agg_factors[0], sub_feats.shape[1] * agg_factors[1])
        #print("resmap", resmap.shape, "ok", ok.shape)
        for i in xrange(resmap.shape[0]):
            for j in xrange(resmap.shape[1]):
                score = resmap[i,j]
                if score >= th and (ok is None or ok[i,j]):
                    #ix = i * psize[0]
                    #iy = j * psize[1]

                    i_corner = i-sub_kernels.shape[1]//2
                    j_corner = j-sub_kernels.shape[2]//2

                    obj_bb = self.boundingboxes[mixcomp]
                    bb = [(i_corner + obj_bb[0]) * agg_factors[0],
                          (j_corner + obj_bb[1]) * agg_factors[1],
                          (i_corner + obj_bb[2]) * agg_factors[0],
                          (j_corner + obj_bb[3]) * agg_factors[1],
                    ]

                    # Clip to bb_bigger 
                    bb = gv.bb.intersection(bb, bb_bigger)
    
                    #score0 = nn_resmaps[0,i,j]
                    #score1 = nn_resmaps[1,i,j]
                    score0 = score1 = 0
                    #print("Score0/1", score0, score1)

                    conf = score
                    #conf = (score - th) / (top_th - th)
                    #conf = np.clip(conf, 0, 1)
                    #plusscore=resplus[i,j], 
                    dbb = gv.bb.DetectionBB(score=score, score0=score0, score1=score1, box=bb, confidence=conf, scale=factor, mixcomp=mixcomp)

                    if gv.bb.area(bb) > 0:
                        bbs.append(dbb)

        # Let's limit to five per level
        bbs_sorted = self.nonmaximal_suppression(bbs)
        bbs_sorted = bbs_sorted[:5]

        return bbs_sorted, resmap

    def response_map(self, sub_feats, sub_kernels, back, mixcomp, level=0, ok=None):
        sh = sub_kernels.shape
        padding = (sh[1]//2, sh[2]//2, 0)
        #padding = (0,)*3
        bigger = ag.util.zeropad(sub_feats, padding)
        #bigger = probpad(sub_feats, padding, back).astype(np.uint8)
        #bigger = ag.util.pad(edges, (sh[1]//2, sh[2]//2, 0), back_kernel[0,0])
        
        #bigger_minus_back = bigger.copy()

        #for f in xrange(edges.shape[-1]):
        #    pass
            #bigger_minus_back[padding[0]:-padding[0],padding[1]:-padding[1],f] -= back_kernel[0,0,f] 
            #bigger_minus_back[padding[0]:-padding[0],padding[1]:-padding[1],f] -= kernels[mixcomp,...,f]

        res = None

        #print('sub_feats', sub_feats.shape)
        
        # With larger kernels, the fftconvolve is much faster. However,
        # this is not the case for smaller kernels.
        if not self.settings.get('use_background_model', 1):
            kern = sub_kernels[mixcomp]
            a = np.log(kern / (1 - kern))
            from .fast import multifeature_correlate2d 
            res = multifeature_correlate2d(bigger, a)
        elif self.use_alpha and False:
            from .fast import llh
            res = llh(bigger, sub_kernels[mixcomp], (self.small_support[mixcomp] > 0.2).astype(np.uint8))
            #res = llh(bigger, sub_kernels[mixcomp], np.empty((0,0), dtype=np.uint8))
            resplus = None
        else:
            #a = np.log(sub_kernels[mixcomp] / (1-sub_kernels[mixcomp]) * ((1-back) / back))
            #a2 = np.log(sub_kernels[mixcomp] / (1-sub_kernels[mixcomp]) * ((1-back) / back))

            if 1:
                from .fast import multifeature_correlate2d, multifeature_correlate2d_with_mask

                kern = sub_kernels[mixcomp]
                #import pdb; pdb.set_trace()
                #print("BACK", back[0])
                weights = np.log(kern/(1-kern) * ((1-back)/back))

                #import pylab as plt
                #plt.imshow(bigger[...,0], interpolation='nearest')
                #plt.colorbar()
                #plt.show()
                #import sys; sys.exit(0)
                    
                
                if ok is None or True:
                    resplus = multifeature_correlate2d(bigger, np.clip(weights, 0, np.inf))
                    #resminus = multifeature_correlate2d(bigger, np.clip(weights, -np.inf, 0))
                    res = multifeature_correlate2d(bigger, weights) 
                else:
                    resplus = multifeature_correlate2d_with_mask(bigger, np.clip(weights, 0, np.inf), ok.astype(np.uint8))
                    res = multifeature_correlate2d_with_mask(bigger, weights, ok.astype(np.uint8)) 
                    
            elif 1:
                from .fast import multifeature_correlate2d 

                resplus = multifeature_correlate2d(bigger, np.clip(np.log(sub_kernels[mixcomp] / back), 0, np.inf))
                resplus += multifeature_correlate2d(1-bigger, np.clip(np.log((1-sub_kernels[mixcomp]) / (1 - back)), -np.inf, 0))

                #res = multifeature_correlate2d(bigger, a)
                res = multifeature_correlate2d(bigger, np.log(sub_kernels[mixcomp] / back))
                res += multifeature_correlate2d(1-bigger, np.log((1-sub_kernels[mixcomp]) / (1 - back)))

            else:
                areversed = a[::-1,::-1]
                biggo = np.rollaxis(bigger, axis=-1)
                arro = np.rollaxis(areversed, axis=-1)
        
                for f in xrange(sub_feats.shape[-1]):
                    r1 = scipy.signal.fftconvolve(biggo[f], arro[f], mode='valid')
                    if res is None:
                        res = r1
                    else:
                        res += r1

    
            #print("level", level, self.train_mean[level])
            if 0:
                res -= self.train_mean[level]
                res /= self.train_std[level]

                resplus -= self.train_mean[level]
                resplus /= self.train_std[level]
            else:
                assert self.num_mixtures == 1, "Need to standardize!"

            if 0:
                import pylab as plt
                plt.subplot(121)
                plt.imshow(res, interpolation='nearest')
                plt.colorbar()
                plt.subplot(122)
                plt.imshow(resplus, interpolation='nearest')
                plt.colorbar()
                plt.show()

            if 0:
                # Subtract expected log likelihood
                res -= (a * back).sum()

                # Standardize 
                summand = a**2 * (back * (1 - back))
                Z = np.sqrt(summand.sum())
                res /= Z
        return res, resplus

    def nonmaximal_suppression(self, bbs):
        # This one will respect scales a bit more
        bbs_sorted = sorted(bbs, reverse=True)

        overlap_threshold = 0.5

        # Suppress within a radius of H neighboring scale factors
        sf = self.settings['scale_factor']
        H = self.settings.get('scale_suppress_radius', 1)
        i = 1
        lo, hi = 1/(H*sf)-0.01, H*sf+0.01
        while i < len(bbs_sorted):
            # TODO: This can be vastly improved performance-wise
            area_i = gv.bb.area(bbs_sorted[i].box)
            for j in xrange(i):
                overlap = gv.bb.area(gv.bb.intersection(bbs_sorted[i].box, bbs_sorted[j].box))/area_i
                scale_diff = (bbs_sorted[i].scale / bbs_sorted[j].scale)
                if overlap > overlap_threshold and \
                   lo <= scale_diff <= hi: 
                    del bbs_sorted[i]
                    i -= 1
                    break

            i += 1
        return bbs_sorted

    def bounding_box_for_mix_comp(self, k):
        """This returns a bounding box of the support for a given component"""

        # Take the bounding box of the support, with a certain threshold.
        #print("Using alpha", self.use_alpha, "support", self.support)
        if self.use_alpha and self.support is not None:
            supp = self.support[k] 
            supp_axs = [supp.max(axis=1-i) for i in xrange(2)]

            # TODO: Make into a setting
            th = self.settings['bounding_box_opacity_threshold'] # threshold
            # Check first and last value of that threshold
            bb = [np.where(supp_axs[i] > th)[0][[0,-1]] for i in xrange(2)]

            # This bb looks like [(x0, x1), (y0, y1)], when we want it as (x0, y0, x1, y1)
            psize = self.settings['subsample_size']
            ret = (bb[0][0]/psize[0], bb[1][0]/psize[1], bb[0][1]/psize[0], bb[1][1]/psize[1])
            #ret = (bb[0][0], bb[1][0], bb[0][1], bb[1][1])
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
        self._preprocess_pooled_support()
        self._preprocess_kernels()
    
        # Prepare bounding boxes for all mixture model
        self.boundingboxes = np.array([self.bounding_box_for_mix_comp(i) for i in xrange(self.num_mixtures)])
    

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

    # TODO: Experimental
    @property
    def use_basis(self):
        return self.kernel_basis is not None

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
            obj.orig_kernel_size = d['orig_kernel_size']
            obj.kernel_basis = d.get('kernel_basis')
            obj.kernel_basis_samples = d.get('kernel_basis_samples')
            obj.kernel_templates = d.get('kernel_templates')
            obj.use_alpha = d['use_alpha']
            obj.support = d.get('support')

            # TODO: Temporary?
            obj.train_std = d.get('train_std')
            obj.train_mean = d.get('train_mean')

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
        d['mixture'] = self.mixture.save_to_dict(save_affinities=True)
        d['orig_kernel_size'] = self.orig_kernel_size
        d['kernel_templates'] = self.kernel_templates
        d['kernel_basis'] = self.kernel_basis
        d['kernel_basis_samples'] = self.kernel_basis_samples
        d['use_alpha'] = self.use_alpha
        d['support'] = self.support
        d['settings'] = self.settings

        # TODO: Temporary?
        try:
            d['train_std'] = self.train_std
            d['train_mean'] = self.train_mean
        except AttributeError:
            pass
        return d
