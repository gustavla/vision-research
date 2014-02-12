from __future__ import division, absolute_import, print_function
from .real_descriptor import RealDescriptor
from .detector import Detector, BernoulliDetector
import numpy as np
import gv

@Detector.register('real')
class RealDetector(BernoulliDetector):
    DESCRIPTOR = RealDescriptor

    def __init__(self, descriptor, settings={}):
        super(RealDetector, self).__init__(settings['num_mixtures'], descriptor, settings)
        self.settings.update(settings)
        self._preprocess()

    def _load_img(self, fn):
        return gv.img.asgray(gv.img.load_image(fn))

    def train_from_features(self, feats, labels, save=True):
        assert len(feats) == len(labels), '{0} != {1}'.format(len(feats), len(labels))
        labels = np.asarray(labels)
        feats = np.asarray(feats)

        pos_feats = feats[labels==1]
        neg_feats = feats[labels==0]

        #K = self.settings.get('num_mixtures', 1)
        # TODO: Only train one at a time
        K = 1
        if K == 1:
            comp_feats = [pos_feats]
        else:
            from sklearn.mixture import GMM
            mixture = GMM(n_components=K, n_iter=10)
            X = pos_feats.reshape((pos_feats.shape[0], -1))
            mixture.fit(X)

            comps = mixture.predict(X)

            comp_feats = [pos_feats[comps==k] for k in xrange(K)]

        kernel_sizes = []
        svms = []
        for k in xrange(K):
            from sklearn import linear_model
            from sklearn import svm
            #from sklearn.svm import sparse
            k_pos_feats = comp_feats[k]

            k_feats = np.concatenate([neg_feats, k_pos_feats])
            k_labels = np.concatenate([np.zeros(len(neg_feats)), np.ones(len(k_pos_feats))])

            #img = images[0]
            kernel_sizes.append(self.settings['image_size'])
            #self.orig_kernel_size = (img.shape[0], img.shape[1])

            flat_k_feats = k_feats.reshape((k_feats.shape[0], -1))        

            from sklearn import cross_validation

            if 0:
                # Set penalty parameter with leave-out validation
                Cs = np.array([1000.0, 500.0, 100.0, 50.0, 10.0, 5.0, 1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001])
                clfs = []
                the_scores = np.zeros(len(Cs))
                for i, C in enumerate(Cs):
                    clf = svm.LinearSVC(C=C)
                    scores = cross_validation.cross_val_score(clf, flat_k_feats.astype(np.float64), k_labels, cv=5)
                    the_scores[i] = np.mean(scores)
                    clfs.append(clf)
                    print('C', C, 'score', np.mean(scores))

                best_i = np.argmax(the_scores)
                print('BEST C', Cs[best_i])
                C = Cs[best_i]
                #svc = clfs[np.argmax(the_scores)] 

            else:
                C = self.settings.get('penalty_parameter', 1)

            import scipy.sparse

            svc = linear_model.SGDClassifier(loss='hinge', penalty='l2', alpha=C, n_iter=100, shuffle=True)
            #X = scipy.sparse.csr_matrix(flat_k_feats)
            #svc.fit(X, k_labels)
            #svc = svm.LinearSVC(C=C)
            #X = scipy.sparse.csr_matrix(flat_k_feats)
            X = flat_k_feats

            #cl = flat_k_feats.clip(min=0.05, max=0.95)
            #from scipy.special import logit
            #coef = logit(flat_k_feats.mean(0).clip(min=0.05, max=0.95))
            #svc.fit(X, k_labels, coef_init=coef)
            svc.fit(X, k_labels)
            #svc.fit(flat_k_feats.astype(np.float64), k_labels) 
            #svc.fit(flat_k_feats.astype(np.float64), k_labels) 

            svms.append(dict(intercept=svc.intercept_, 
                             weights=svc.coef_.reshape(feats.shape[1:])))
    
        if save:
            self._preprocess()
            self.svms = svms
            self.kernel_sizes = kernel_sizes

        return svms, kernel_sizes


    def train_from_images(self, image_filenames, labels):
        assert len(image_filenames) == len(labels)

        images = [self._load_img(fn) for fn in image_filenames]
        feats = np.asarray([self.extract_spread_features(img) for img in images])

        return self.train_from_features(feats, labels)

    def train_from_image_data(self, images, labels):
        assert len(images) == len(labels)

        feats = np.asarray([self.extract_spread_features(img) for img in images])

        return self.train_from_features(feats, labels)


    def detect_coarse_single_factor(self, img, factor, mixcomp, img_id=0, cascade=True, discard_weak=False, farming=False, return_bounding_boxes=True, strides=(1, 1), *args, **kwargs):
        bb_bigger = (0, 0, img.shape[0], img.shape[1])

        img_resized = gv.img.resize_with_factor_new(gv.img.asgray(img), 1/factor) 

        spread_feats = self.extract_spread_features(img_resized)

        bbs, resmap = self._detect_coarse_at_factor(spread_feats, 
                                                    factor, 
                                                    mixcomp, 
                                                    bb_bigger, 
                                                    cascade=cascade, 
                                                    discard_weak=discard_weak, 
                                                    farming=farming,
                                                    return_bounding_boxes=return_bounding_boxes,
                                                    strides=strides)

        final_bbs = bbs

        return final_bbs, resmap, None, spread_feats, img_resized

    def weights(self, mixcomp):
        return self.svms[mixcomp]['weights']

    def keypoint_mask(self, mixcomp):
        kp_only_weights = np.ones(self.weights_shape(mixcomp))
        return kp_only_weights

    def _response_map(self, feats, mixcomp, strides=(1, 1)):
        sh = self.svms[mixcomp]['weights'].shape
        pmult = self.settings.get('padding_multiple_of_object', 0.5)
        padding = (int(sh[0]*pmult), int(sh[1]*pmult), 0)

        if min(feats.shape[:2]) < 2:
            return np.array([]), None, padding

        bigger = gv.ndfeature.zeropad(feats, padding)


        from .fast import multifeature_real_correlate2d
        #index = 26 
        #res = multifeature_correlate2d(bigger[...,index:index+1], weights[...,index:index+1].astype(np.float64)) 

        weights = self.svms[mixcomp]['weights']
        if bigger.shape[0] < weights.shape[0] or bigger.shape[1] < weights.shape[1]:
            return np.zeros((0, 0, 0)), None, padding 

        res = self.svms[mixcomp]['intercept'] + \
              multifeature_real_correlate2d(bigger.astype(np.float64), weights, strides=strides)

        lower, upper = gv.ndfeature.inner_frame(bigger, (sh[0]/2, sh[1]/2))
        res = gv.ndfeature(res, lower=lower, upper=upper)

        return res, bigger, padding

    @property
    def subsample_size(self):
        return self.descriptor.subsample_size

    def _detect_coarse_at_factor(self, feats, factor, mixcomp, bb_bigger, cascade=True, farming=False, 
                                 discard_weak=False, return_bounding_boxes=True, strides=(1, 1)):
        # Get background level
        resmap, bigger, padding = self._response_map(feats, mixcomp, strides=strides)

        if np.min(resmap.shape) <= 1:
            return [], resmap

        kern = self.svms[mixcomp]['weights']


        # TODO: Decide this in a way common to response_map
        sh = kern.shape
        sh0 = kern.shape

        # Get size of original kernel (but downsampled)
        full_sh = self.kernel_sizes[mixcomp]
        psize = self.subsample_size
        sh2 = sh
        sh = (full_sh[0]//psize[0], full_sh[1]//psize[1])

        import scipy.stats
        if farming:
            percentile = 75 
        else:
            percentile = 75
        th = scipy.stats.scoreatpercentile(resmap.ravel(), percentile) 
        top_th = 200.0
        bbs = []

        agg_factors = tuple([psize[i] * factor for i in xrange(2)])
        agg_factors2 = tuple([factor for i in xrange(2)])
        #bb_bigger = (0.0, 0.0, feats.shape[0] * agg_factors[0], feats.shape[1] * agg_factors[1])
        bbs_sorted = [] 
        if return_bounding_boxes:
            for i in xrange(resmap.shape[0]):
                for j in xrange(resmap.shape[1]):
                    score = resmap[i,j]
                    if score >= th:
                        X = bigger[i:i+sh0[0], j:j+sh0[1]].copy()
                        ok = True
            
                        # Cascade
                        if cascade and 0:
                            cur_score = score
                            import itertools as itr
                            for cas_i, cas in enumerate(self.extra['cascades']):
                                svm = cas['svms'][mixcomp]
                                threshold = cas['th']

                                if cur_score < threshold:
                                    ok = False
                                    break

                                score0 = float(svm['intercept'] + np.sum(svm['weights'] * X))

                                #if score0 < threshold:
                                    #break
                                #else:
                                score = score0 + 10 * (cas_i + 1)
                                cur_score = score


                        if discard_weak and not ok:
                            continue

                        conf = score
                        pos = resmap.pos((i, j))
                        #lower = resmap.pos((i + self.boundingboxes2[mixcomp][0]))
                        bb = ((pos[0] * agg_factors2[0] + self.boundingboxes2[mixcomp][0] * agg_factors[0]),
                              (pos[1] * agg_factors2[1] + self.boundingboxes2[mixcomp][1] * agg_factors[1]),
                              (pos[0] * agg_factors2[0] + self.boundingboxes2[mixcomp][2] * agg_factors[0]),
                              (pos[1] * agg_factors2[1] + self.boundingboxes2[mixcomp][3] * agg_factors[1]))

                        bb = gv.bb.intersection(bb, bb_bigger)

                        index_pos = (i-padding[0], j-padding[1])

                        dbb = gv.bb.DetectionBB(score=score, box=bb, index_pos=index_pos, confidence=conf, scale=factor, mixcomp=mixcomp, bkgcomp=0, X=X)

                        if gv.bb.area(bb) > 0:
                            bbs.append(dbb)

            # Let's limit to five per level
            if farming:
                bbs_sorted = sorted(bbs, reverse=True)
                bbs_sorted = bbs_sorted[:100]
            else:
                bbs_sorted = self.nonmaximal_suppression(bbs)
                bbs_sorted = bbs_sorted[:15]

        return bbs_sorted, resmap
        

    @classmethod
    def load_from_dict(cls, d):
        try:
            descriptor_cls = cls.DESCRIPTOR.getclass(d['descriptor_name'])
            if descriptor_cls is None:
                raise Exception("The descriptor class {0} is not registered".format(d['descriptor_name'])) 
            descriptor = descriptor_cls.load_from_dict(d['descriptor'])
            obj = cls(descriptor, d['settings'])
            #obj.weights = d['weights']
            obj.svms = d['svms']
            obj.extra = d['extra']
            obj.support = d.get('support')
            #obj.orig_kernel_size = d.get('orig_kernel_size')
            obj.kernel_sizes = d.get('kernel_sizes')

            obj._preprocess()
            
            return obj
        except KeyError as e:
            # TODO: Create a new exception for these kinds of problems
            raise Exception("Could not reconstruct class from dictionary. Missing '{0}'".format(e))

    def save_to_dict(self):
        d = {}
        d['descriptor_name'] = self.descriptor.name
        d['descriptor'] = self.descriptor.save_to_dict()
        #d['weights'] = self.weights
        d['extra'] = self.extra
        d['svms'] = self.svms
        #d['orig_kernel_size'] = self.orig_kernel_size
        d['kernel_sizes'] = self.kernel_sizes
        d['support'] = self.support
        d['settings'] = self.settings

        return d
