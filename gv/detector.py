
from __future__ import division
from patch_dictionary import PatchDictionary
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
        
    It uses the PatchDictionary as features, and then runs a mixture model on top of that.
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
        mixture.run_EM(1e-6, self.settings['min_probability'])
        
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
    
            # Or stretch them out?
            m[m == m0] = 0.0
            m[m == m1] = 1.0
            # What this is essentially doing is boosting everything a bit, nothing is
            # completely opaque, which might be the source of the boost.
            # If we add "alpha *" before m, then it's as bad as the other method (and
            # with very similar results)
            #self.kernels[...,f] = 0.5 * (1-alpha) + m#self.mixture.templates[...,f]# / alpha

        # TODO: Putting *3 here makes it stop favor background. UNDERSTAND!
        #self.kernels *= 1 

        eps = self.settings['min_probability']
        self.kernels = np.clip(self.kernels, eps, 1-eps)

        #self.mixture.templates = np.clip(self.mixture.templates, 0.5, 1.0)
        self.log_kernels = np.log(self.kernels)
        self.log_invkernels = np.log(1.0-self.kernels)
        self.log_kernel_ratios = np.log(self.kernels / (1.0 - self.kernels))

    def extract_features(self, image):
        edges = ag.features.bedges_from_image(image, **self.patch_dict.bedges_settings())
        small = self.patch_dict.extract_pooled_parts(edges)
        return small

    def prepare_kernels(self, image, mixcomp):
        # Convert image to our feature space representation
        edges = ag.features.bedges_from_image(image, **self.patch_dict.bedges_settings())
        small = self.patch_dict.extract_pooled_parts(edges)

        # Figure out background probabilities of this image
        back = np.zeros(self.log_kernels.shape[1:]) 
        for f in xrange(small.shape[-1]):
            back[...,f] = small[...,f].sum() / np.prod(small.shape[:2])

        print "Backs: {0} (std: {1}) [{2}, {3}]".format(back.mean(), back.std(), back.min(), back.max())

        print "Most ubiquitous:", np.argmax(back)

        #back[...] = 0.05

        back = np.clip(back, 0.05, 0.95)


    
        #print back.shape
        #print back
        #self.log_back = np.log(back)
        #self.log_invback = np.log(1.0 - back)

        # Create kernels just for this case
        kernels = self.kernels.copy()

        ss = self.small_support[mixcomp].copy()
        ss *= 2 
        ss = np.clip(ss, 0, 1)
        #kernels[mixcomp] *= 1.67
        #kernels[mixcomp] *= 1.10
        #kernels[mixcomp] *= 1.38
        print 'max kernels', kernels[mixcomp].max()
    
        #total = (kernels[mixcomp] - back).sum()
        #print 'TOTAL', total

        score_lower = (np.log(1.0 - kernels[mixcomp]) - np.log(1.0 - back)).sum()
        score = np.inf

        lower = 0.0001
        higher = 10.0
        middle = 1.0 
    
        limit = 0
        while np.fabs(score_lower - score) > 10.0 and limit < 30:
            limit += 1
            middle = (lower + higher) / 2.0
            kerny = kernels[mixcomp].copy()
            kerny *= middle
            for f in xrange(small.shape[-1]):
                kerny[...,f] = np.clip((1-ss) * back[0,0,f] + ss * kerny[...,f], 0.05, 0.95)


            score_lower = (np.log(1.0 - kerny) - np.log(1.0 - back)).sum()
            score = (back * (np.log(kerny) - np.log(back)) + \
                     (1-back) * (np.log(1.0 - kerny) - np.log(1.0 - back))).sum()

            print lower, higher, score_lower, score

            if score < score_lower:
                lower = middle
            else:
                higher = middle

        kernels *= middle
        print 'middle', middle
        for f in xrange(small.shape[-1]):
            kernels[mixcomp,...,f] = np.clip((1-ss) * back[0,0,f] + ss * kernels[mixcomp,...,f], 0.05, 0.95)
        
        print "MAX KERNELS", kernels[mixcomp].max()
        
            
        #for x in xrange(kernels.shape[1]):
            #for y in xrange(kernels.shape[2]):
                #kernels[mixcomp,x,y] = kernels[mixcomp,x,y] / kernels[mixcomp,x,y].sum()

        print '----'
        #print np.prod(k/(1-k))
        #print np.prod(kernels[mixcomp].sum()

        score2 = (np.log(kernels[mixcomp]) - np.log(back)).sum()
        score3 = (back * (np.log(kernels[mixcomp]) - np.log(back)) + \
                  (1-back) * (np.log(1.0 - kernels[mixcomp]) - np.log(1.0 - back)))
        print 'shape', score3.shape
        score3 = score3.sum()

        print "Back score", score_lower
        print "Front score", score2
        print "Middle score", score3
   
    

        return back, kernels, small

    def response_map(self, image, mixcomp):
        """Retrieves log-likelihood response on 'image' (no scaling done)"""

        back, kernels, small = self.prepare_kernels(image, mixcomp)

        sh = kernels.shape
        bigger = ag.util.zeropad(small, (sh[1]//2, sh[2]//2, 0))
        #bigger = ag.util.zeropad(small, (sh[1], sh[2], 0))
        #bigger = probpad(small, (sh[1]//2, sh[2]//2, 0), back[0,0])

        res = None
        for k in [mixcomp]:#xrange(self.num_mixtures):
        #for k in xrange(self.mixture.num_mix):
            if 1:
                # TODO: Place outside of forloop (k) !
                #bigger = ag.util.zeropad(small, (sh[1], sh[2], 0))
                from masked_convolve import masked_convolve
                # TODO: Missing constant now
                #r1 = masked_convolve(bigger, self.log_kernel_ratios[k])
                #r2 = 0.0
                r1 = masked_convolve(bigger, np.log(kernels[k]))
                r2 = masked_convolve(1-bigger, np.log(1.0 - kernels[k]))
                r3 = masked_convolve(1-bigger, -np.log(1.0 - back))
                r4 = masked_convolve(bigger, -np.log(back))
                print 'TOP-LEFTS', r1[0,0], r2[0,0], r3[0,0], r4[0,0]
                #print r1.sum(), r2.sum(), r3.sum(), r4.sum()
                #res += r1 + r2
                if res is None:
                    res = r1 + r2 + r3 + r4
                else:
                    res += r1 + r2 + r3 + r4

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
            
                    print r3, r4

                    if res is None:
                        res = r1 + r2 + r3 + r4
                    else:
                        res += r1 + r2 + r3 + r4

        ksh = sh[1:3]

        score_lower = (np.log(1.0 - kernels[mixcomp]) - np.log(1.0 - back)).sum()
        score_upper = (np.log(kernels[mixcomp]) - np.log(back)).sum()
        print "Back score", score_lower
        print "Front score", score_upper

        # Normalize
        print 'ksh', ksh
        print sh[1:]
        print 'res', res.shape
        print 'small', small.shape
        print 'bigger', bigger.shape
        densities = np.empty(res.shape)

        ss = self.small_support[mixcomp].copy()
        ss *= 2 
        ss = np.clip(ss, 0, 1)

        print "Top-left (pre-normalization):", res[0,0]

        for x in xrange(res.shape[0]):
            for y in xrange(res.shape[1]):
                density = (ss.reshape(ss.shape + (1,)) * bigger[x:x + ksh[0], y:y + ksh[1]]).sum() / (ss.sum() * sh[-1]) #np.prod(sh[1:])
                #print density
                #res[x,y] = (res[x,y] + 0.001) / max(density, 0.0000000000000000001)
                #res[x,y] = (res[x,y] - score_lower) - density * (score_upper - score_lower)
                # Doesn't matter if standardizing
                res[x,y] -= score_lower
                densities[x,y] = density

        print 'TOP', np.unravel_index(res.argmax(), res.shape)

        if 0:
            import pylab as plt
            plt.imshow(densities, interpolation='nearest')
            plt.colorbar()
            plt.show()

        print "Top-left:", res[0,0]
        return res, small

    def resize_and_detect(self, img, mixcomp, factor=1.0):
        img_resized = gv.img.resize(img, factor)
        x, img_feat = self.response_map(img_resized, mixcomp)
        return x, img_feat, img_resized

    def detect_coarse_unfiltered_at_scale(self, img, factor, mixcomp):
        x, small, img_resized = self.resize_and_detect(img, mixcomp, factor)

        #import ipdb; ipdb.set_trace()

        # Frst pick 
        th = -37000#-35400 
        #th = -36000
        th = -35400 + 70 - 2500
        #th = -36040 - 1
        th = 0.0#15
        #th = -10002.0 
        #th = 800.0
        th = 750.0
        GET_ONE = True#False 
        if GET_ONE:
            th = -10000000000

        bbs = []
        
        xx = x
        xx = (x - x.mean()) / x.std()

        if 0:
            import pylab as plt
            plt.hist(xx.flatten(), 50)
            plt.show()

        print 'feature activity:', small.sum() / np.prod(small.shape)
        print 'x max', x.max()

        edges = (0.0, 0.0, img.shape[0], img.shape[1])

        for i in xrange(x.shape[0]):
            for j in xrange(x.shape[1]):
                if xx[i,j] > th:
                    #print factor, xx[i,j]
                    pooling_size = self.patch_dict.settings['pooling_size']
                    ix = i * pooling_size[0]
                    iy = j * pooling_size[1]
                    bb = self.bounding_box_at_pos((ix, iy), mixcomp)
                    # TODO: The above function should probably return in this coordinates
                    bb = tuple([bb[k] / factor for k in xrange(4)])
                    # Clip to edges
                    bb = gv.bb.intersection(bb, edges)
                    score = xx[i,j]
                    dbb = gv.bb.DetectionBB(score=score, box=bb, confidence=np.clip((score-th)/2.0, 0, 1))

                    if gv.bb.area(bb) > 0:
                        bbs.append(dbb)

        if GET_ONE:
            bbs.sort(reverse=True)
            bbs = bbs[:1]
    
        return bbs, xx, small

    def detect_coarse(self, img, fileobj=None):
        bbs = []
        df = 0.05
        #df = 0.1
        factors = np.arange(0.3, 1.0+0.01, df)
        for factor in factors:
            for mixcomp in xrange(self.num_mixtures):
                bbsthis, _, _ = self.detect_coarse_unfiltered_at_scale(img, factor, mixcomp)
                bbs += bbsthis

        # Do NMS here
        final_bbs = self.nonmaximal_suppression(bbs)
        
        # Mark corrects here
        if fileobj is not None:
            self.label_corrects(final_bbs, fileobj)
        return final_bbs

    def detect_coarse_single_component(self, img, mixcomp, fileobj=None):
        df = 0.05
        #df = 0.1
        factors = np.arange(0.3, 1.0+0.01, df)

        bbs = []
        for factor in factors:
            print "Running factor", factor
            bbsthis, _, _ = self.detect_coarse_unfiltered_at_scale(img, factor, mixcomp)
            print "found {0} bounding boxes".format(len(bbsthis))
            bbs += bbsthis
    
        # Do NMS here
        final_bbs = self.nonmaximal_suppression(bbs)

        # Mark corrects here
        if fileobj is not None:
            self.label_corrects(final_bbs, fileobj)
        return final_bbs

    def nonmaximal_suppression(self, bbs):
        bbs_sorted = sorted(bbs, reverse=True)

        overlap_threshold = 0.5

        print 'bbs length', len(bbs_sorted)
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
        print 'bbs length', len(bbs_sorted)
        return bbs_sorted
    
    def bounding_box_for_mix_comp(self, k):
        """This returns a bounding box of the support for a given component"""
        # Take the bounding box of the support, with a certain threshold.
        supp = self.support[k] 
        supp_axs = [supp.max(axis=1-i) for i in xrange(2)]

        # TODO: Make into a setting
        th = self.settings['bounding_box_opacity_threshold'] # threshold
        # Check first and last value of that threshold
        bb = [np.where(supp_axs[i] > th)[0][[0,-1]] for i in xrange(2)]

        # This bb looks like [(x0, x1), (y0, y1)], when we want it as (x0, y0, x1, y1)
        return (bb[0][0], bb[1][0], bb[0][1], bb[1][1])

    def bounding_box_at_pos(self, pos, mixcomp):
        supp_size = self.support.shape[1:]
        bb = self.bounding_box_for_mix_comp(mixcomp)

        pos0 = [pos[i]-supp_size[i]//2 for i in xrange(2)]
        return (pos0[0]+bb[0],   
                pos0[1]+bb[1], 
                pos0[0]+bb[2], 
                pos0[1]+bb[3])


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
