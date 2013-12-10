from __future__ import absolute_import, division
import random
import copy
import amitgroup as ag
import numpy as np
import gv
import math
import itertools as itr
import scipy
import scipy.stats
from .binary_descriptor import BinaryDescriptor

def convert_partprobs_to_feature_vector(partprobs, tau=0.0):
    X_dim_0 = partprobs.shape[0]
    X_dim_1 = partprobs.shape[1]
    num_parts = partprobs.shape[2] - 1
    d = 0.0
    mx = 0.0
    feats = np.zeros((X_dim_0, X_dim_1, num_parts), dtype=np.uint8)

    for i in xrange(X_dim_0):
        for j in xrange(X_dim_1):
            m = partprobs[i,j].argmax()
            mx = partprobs[i,j,m]
            if m != 0:
                if mx > -150:
                    d = partprobs[i,j,m] - 20
                    for f in xrange(1,num_parts+1):
                        if partprobs[i,j,f] >= d:
                            feats[i,j,f-1] = 1 
                else:
                    feats[i,j,m-1] = 1

    
    return feats

@BinaryDescriptor.register('polarity-parts')
class PolarityPartsDescriptor(BinaryDescriptor):
    def __init__(self, patch_size, num_parts, settings={}):
        self.patch_size = patch_size
        self._num_parts = num_parts 

        self.parts = None
        self.unspread_parts = None
        self.unspread_parts_padded = None
        self.visparts = None

        self.settings = {}
        self.settings['patch_frame'] = 1
        self.settings['threshold'] = 4 
        self.settings['threaded'] = False 
        self.settings['samples_per_image'] = 500 
        self.settings['min_probability'] = 0.005
        self.settings['strides'] = 1
    
        # TODO
        self.extra = {}

        # Or maybe just do defaults?
        # self.settings['bedges'] = {}
        self.settings['bedges'] = dict(k=5, radius=1, minimum_contrast=0.05, contrast_insensitive=True, max_edges=2)
        self.settings.update(settings)

    @property
    def num_features(self):
        return self._num_parts

    # TODO: Remove this one
    @property
    def num_parts(self):
        return self._num_parts

    @property
    def subsample_size(self):
        return self.settings['subsample_size']

    def _get_patches(self, filename):
        samples_per_image = self.settings['samples_per_image']
        fr = self.settings['patch_frame']
        the_patches = []
        the_unspread_patches = []
        the_unspread_patches_padded = []
        the_originals = []
        ag.info("Extracting patches from", filename)
        #edges, img = ag.features.bedges_from_image(f, k=5, radius=1, minimum_contrast=0.05, contrast_insensitive=False, return_original=True, lastaxis=True)
        setts = self.settings['bedges'].copy()
        radius = setts['radius']
        setts2 = setts.copy()
        setts['radius'] = 0

        # LEAVE-BEHIND
        img = gv.img.load_image(filename)
        img = gv.img.asgray(img)
        inv_img = 1 - img
        if 0:
            unspread_edges = ag.features.bedges(img, **setts)
            inv_unspread_edges = ag.features.bedges(inv_img, **setts)
        else:
            unspread_edges = self._extract_edges(img)
            inv_unspread_edges = self._extract_edges(inv_img)

        unspread_edges_padded = ag.util.zeropad(unspread_edges, (radius, radius, 0))
        inv_unspread_edges_padded = ag.util.zeropad(inv_unspread_edges, (radius, radius, 0))

        # Spread the edges
        edges = ag.features.bspread(unspread_edges, spread=setts['spread'], radius=radius)
        inv_edges = ag.features.bspread(inv_unspread_edges, spread=setts['spread'], radius=radius)

        #setts2['minimum_contrast'] *= 2
        #edges2 = ag.features.bedges(img, **setts2)
        #inv_edges2 = ag.features.bedges(inv_img, **setts2)
        
        #s = self.settings['bedges'].copy()
        #if 'radius' in s:
        #    del s['radius']
        #edges_nospread = ag.features.bedges_from_image(filename, radius=0, **s)

        # How many patches could we extract?
        w, h = [edges.shape[i]-self.patch_size[i]+1 for i in xrange(2)]

        # TODO: Maybe shuffle an iterator of the indices?
        indices = list(itr.product(xrange(w-1), xrange(h-1)))
        random.shuffle(indices)
        i_iter = iter(indices)

        for sample in xrange(samples_per_image):
            for tries in xrange(20):
                #x, y = random.randint(0, w-1), random.randint(0, h-1)
                x, y = i_iter.next()
                selection = [slice(x, x+self.patch_size[0]), slice(y, y+self.patch_size[1])]
                selection_padded = [slice(x, x+radius*2+self.patch_size[0]), slice(y, y+radius*2+self.patch_size[1])]
                # Return grayscale patch and edges patch
                edgepatch = edges[selection]
                inv_edgepatch = inv_edges[selection]
                #edgepatch2 = edges2[selection]
                #inv_edgepatch2 = inv_edges2[selection]
                #edgepatch_nospread = edges_nospread[selection]

                # The inv edges could be incorproated here, but it shouldn't be that different.
                if fr == 0:
                    avg = edgepatch.mean()
                else:
                    avg = edgepatch[fr:-fr,fr:-fr].mean()

                if self.settings['threshold'] <= avg <= self.settings.get('max_threshold', np.inf): 
                    #the_patches.append(np.asarray([edgepatch, inv_edgepatch, edgepatch2, inv_edgepatch2]))
                    the_patches.append(np.asarray([edgepatch, inv_edgepatch]))
                    #the_patches.append(edgepatch_nospread)
    
                    #the_unspread_patch = unspread_edges[selection]
                    #the_unspread_patches.append(the_unspread_patch)

                    #the_unspread_patch_padded = unspread_edges_padded[selection_padded]
                    #the_unspread_patches_padded.append(the_unspread_patch_padded)
        
                    # The following is only for clearer visualization of the 
                    # patches. However, normalizing like this might be misleading
                    # in other ways.
                    vispatch = img[selection]
                    inv_vispatch = inv_img[selection]

                    span = vispatch.min(), vispatch.max() 
                    if span[1] - span[0] > 0:
                        vispatch = (vispatch-span[0])/(span[1]-span[0])

                    inv_span = inv_vispatch.min(), inv_vispatch.max() 
                    if inv_span[1] - inv_span[0] > 0:
                        inv_vispatch = (inv_vispatch-inv_span[0])/(inv_span[1]-inv_span[0])

                    #the_originals.append(np.asarray([vispatch, inv_vispatch]*2))
                    the_originals.append(np.asarray([vispatch, inv_vispatch]))
                    break

        return the_patches, the_originals

    def random_patches_from_images(self, filenames):
        raw_patches = []
        raw_unspread_patches = []
        raw_unspread_patches_padded = []
        raw_originals = [] 

        # TODO: Have an amitgroup / vision-research setting for "allow threading"
        if 0:
            if True:#self.settings['threaded']:
                from multiprocessing import Pool
                p = Pool(7) # Should not be hardcoded
                mapfunc = p.map
            else:
                mapfunc = map

        mapfunc = map

        ret = mapfunc(self._get_patches, filenames)

        for patches, originals in ret:
            raw_patches.extend(patches)
            raw_originals.extend(originals) 

        raw_patches = np.asarray(raw_patches)
        raw_originals = np.asarray(raw_originals)
        return raw_patches, raw_originals

    def bedges_settings(self):
        return self.settings['bedges']

    def train_from_images(self, filenames):
        raw_patches, raw_originals = self.random_patches_from_images(filenames)
        self.P = raw_patches.shape[1]

        if len(raw_patches) == 0:
            raise Exception("No patches found, maybe your thresholds are too strict?")
        # Also store these in "settings"

        mixtures = []
        llhs = []
        from gv.polarity_bernoulli_mm import PolarityBernoulliMM

        mixture = PolarityBernoulliMM(n_components=self._num_parts, n_iter=6, random_state=0, min_probability=self.settings['min_probability'])
        mixture.fit(raw_patches.reshape(raw_patches.shape[:2] + (-1,)))

        ag.info("Done.")

        comps = mixture.mixture_components()
        counts = np.bincount(comps[:,0], minlength=self.num_features)
        self.extra['counts'] = counts

        print counts
        print 'Total', np.sum(counts)
        from scipy.stats.mstats import mquantiles
        print mquantiles(counts)


        llhs = np.zeros(len(raw_patches))
        # Now, sort out low-likelihood patches
        for i, patch in enumerate(raw_patches):
            patch0 = patch[0]
            model = mixture.means_[tuple(comps[i])].reshape(patch0.shape)
            llh = (patch0 * np.log(model)).sum() + ((1 - patch0) * np.log(1 - model)).sum()
            llhs[i] = llh

        means = []
        sigmas = []
        for f in xrange(self.num_features):
            model = mixture.means_[f,0] 
            mu = np.sum((model * np.log(model)) + (1 - model) * np.log(1 - model))
            a = np.log(model / (1 - model))
            sigma = np.sqrt((a**2 * model * (1 - model)).sum())

            means.append(mu)
            sigmas.append(sigma)
        means = np.asarray(means)
        sigmas = np.asarray(sigmas)

        #self.extra['means'] = np.asarray(means)
        #self.extra['stds'] = np.asarray(sigmas)

        self.extra['means'] = scipy.ndimage.zoom(means, 2, order=0)
        self.extra['stds'] = scipy.ndimage.zoom(sigmas, 2, order=0)


        means2 = []
        sigmas2 = []
        
        II = comps[:,0]

        for f in xrange(self.num_features):
            X = llhs[II == f]
            mu = X.mean()
            sigma = X.std()

            means2.append(mu)
            sigmas2.append(sigma)

        #self.extra['means_emp'] = np.asarray(means2)
        #self.extra['stds_emp'] = np.asarray(sigmas2)

        self.extra['means_emp'] = scipy.ndimage.zoom(means2, 2, order=0)
        self.extra['stds_emp'] = scipy.ndimage.zoom(sigmas2, 2, order=0)

        II = comps[:,0]

        #self.extra['comps'] = comps

        #self.extra['llhs'] = llhs

        if 0:
            keepers = np.zeros(llhs.shape, dtype=np.bool)

            for f in xrange(self.num_features):
                llhs_f = llhs[II == f] 
                th_f = scipy.stats.scoreatpercentile(llhs_f, 80)

                keepers[(II == f) & (llhs >= th_f)] = True
        else:
            keepers = np.ones(llhs.shape, dtype=np.bool)

        #th = scipy.stats.scoreatpercentile(llhs, 75)
        #keepers = llhs >= th

        raw_patches = raw_patches[keepers]
        raw_originals = raw_originals[keepers]
        #raw_originals = list(itr.compress(raw_originals, keepers))
        comps = comps[keepers]

        ag.info("Pruned")

        self.parts = mixture.means_.reshape((mixture.n_components, self.P) + raw_patches.shape[2:])
        self.orientations = None 
        #self.unspread_parts = self.unspread_parts[II]
        #self.unspread_parts_padded = self.unspread_parts_padded[II]
        self.unspread_parts = None
        self.unspread_parts_padded = None
        #self.visparts = None#self.visparts[II]

        #visparts = mixture.remix(raw_originals)
        #aff = np.asarray(self.affinities)
        #self.visparts = np.asarray([np.average(data, axis=0, weights=aff[:,m]) for m in xrange(self.num_mix)])
        self.visparts = np.zeros((mixture.n_components,) + raw_originals.shape[2:])
        counts = np.zeros(mixture.n_components)
        for n in xrange(raw_originals.shape[0]):
            f, polarity = comps[n]# np.unravel_index(np.argmax(mixture.q[n]), mixture.q.shape[1:])
            self.visparts[f] += raw_originals[n,polarity]
            counts[f] += 1
        self.visparts /= counts[:,np.newaxis,np.newaxis]

        #self.visparts = np.asarray([self.visparts, 1 - self.visparts])
        #self.visparts = np.rollaxis(self.visparts, 1)
        #self.visparts = self.visparts.reshape((self.visparts.shape[0] * 2,) + self.visparts.shape[2:])

        #self.extra['originals'] = [self.extra['originals'][ii] for ii in II]

        #self.weights = mixture.weights_

        #self.extra['originals'] = originals
        weights = mixture.weights_

        #II = np.argsort(weights.max(axis=1))
        #II = np.arange(self.parts.shape[0])

        #self.extra['originals'] = [originals[ii] for ii in II]
        #self.extra['originals'] = [self.extra['originals'][ii] for ii in II]

        self._num_parts = self.parts.shape[0]

        self.parts = self.parts.reshape((self.parts.shape[0] * self.P,) + self.parts.shape[2:])

        order_single = np.argsort(means / sigmas)
        new_order_single = []
        for f in order_single: 
            part = self.parts[2*f] 
            sh = part.shape
            p = part.reshape((sh[0]*sh[1], sh[2]))
            
            pec = p.mean(axis=0)

            N = np.sum(p * np.log(p/pec) + (1-p)*np.log((1-p)/(1-pec)))
            D = np.sqrt(np.sum(np.log(p/pec * (1-pec)/(1-p))**2 * p * (1-p)))

            ok = (N/D > 1) and counts[f] > 50
              
            if ok: 
                new_order_single.append(f)

        order_single = np.asarray(new_order_single)

        assert len(order_single) > 0, "No edges kept! Something probably went wrong"

        self._num_parts = len(order_single)
        print 'num_parts', self._num_parts

        order = np.zeros(self.num_features * 2, dtype=int) 
        for f in xrange(self.num_features):
            order[2*f] = order_single[f]*2
            order[2*f+1] = order_single[f]*2+1
        II = order

        # Require at least 20 originals
        #II = filter(lambda ii: counts[ii] >= 20, II)

        # Resort everything
        self.extra['means'] = self.extra['means'][II]
        self.extra['stds'] = self.extra['stds'][II]
        self.extra['means_emp'] = self.extra['means_emp'][II]
        self.extra['stds_emp'] = self.extra['stds_emp'][II]
        #self.extra['means_double']

        self.parts = self.parts[II]
        self.visparts = self.visparts[order_single]

        originals = []
        for i in order_single:
            j = weights[i].argmax()

            # You were here: This line is not correct
            #import pdb; pdb.set_trace()

            II0 = np.where(comps[:,0] == i)[0]
            II1 = comps[II0,1]

            ims = np.asarray([raw_originals[tuple(ij)] for ij in itr.izip(II0, II1)])
            #ims = raw_originals[comps[:,0] == i,comps[:,1]].copy()
            originals.append(ims[:50])
        self.extra['originals'] = originals

        #scores = scores[II]
        #counts = counts[II]

        #self.extra['scores'] = scores
        #self.extra['counts'] = counts
        #self.extra['weights'] = weights[II]

        #for f in xrange(self.num_parts):
            #part = mixture.means_[f]
            #sh = part.shape
            #p = part.reshape((sh[0]*sh[1], 

        # Reject weak parts
        if 0:
            counts = np.bincount(mixture.mixture_components(), minlength=self.num_parts)
            scores = np.empty(self.num_parts) 
            for i in xrange(self.num_parts):
                part = mixture.templates[i]
                sh = part.shape
                p = part.reshape((sh[0]*sh[1], sh[2]))
                
                pec = p.mean(axis=0)
            
                N = np.sum(p * np.log(p/pec) + (1-p)*np.log((1-p)/(1-pec)))
                D = np.sqrt(np.sum(np.log(p/pec * (1-pec)/(1-p))**2 * p * (1-p)))
                # Old:
                #D = np.sqrt(np.sum(np.log(p/(1-p))**2 * p * (1-p)))

                scores[i] = N/D 

                # Require at least 20 occurrences
                #if counts[i] < 5:
                    #scores[i] = 0

            # Only keep with a certain score
            if not self.settings['bedges']['contrast_insensitive']:

                visparts = mixture.remix(raw_originals)
            else:
                visparts = np.empty((self.num_parts,) + raw_originals.shape[1:])

                self.extra['originals'] = []
            
                # Improved visparts
                comps = mixture.mixture_components()
                for i in xrange(self.num_parts):
                    ims = raw_originals[comps == i].copy()

                    self.extra['originals'].append(ims)

                    # Stretch them all out
                    #for j in xrange(len(ims)):
                        #ims[j] = (ims[j] - ims[j].min()) / (ims[j].max() - ims[j].min())

                    # Now, run a GMM with NM components on this and take the most common
                    NM = 2

                    from sklearn.mixture import GMM
                    gmix = GMM(n_components=NM)
                    gmix.fit(ims.reshape((ims.shape[0], -1)))

                    visparts[i] = gmix.means_[gmix.weights_.argmax()].reshape(ims.shape[1:])

            # Unspread parts
            #unspread_parts_all = mixture.remix(raw_unspread_patches) 
            #unspread_parts_padded_all = mixture.remix(raw_unspread_patches_padded) 

            # The parts to keep
            ok = (scores > 1) & (counts >= 10)

            if 'originals' in self.extra:
                self.extra['originals'] = list(itr.compress(self.extra['originals'], ok))

            scores = scores[ok]
            counts = counts[ok]
            
            self.parts = mixture.templates[ok]
            #self.unspread_parts = unspread_parts_all[ok]
            #self.unspread_parts_padded = unspread_parts_padded_all[ok]
            self.visparts = visparts[ok]
            self._num_parts = self.parts.shape[0]
            
            # Update num_parts
            
            # Store the stuff in the instance
            #self.parts = mixture.templates
            #self.visparts = mixture.remix(raw_originals)


            # Sort the parts according to orientation, for better diagonistics
            if 1:
                E = self.parts.shape[-1]
                E = self.parts.shape[-1]
                ang = np.array([[0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, 1]])
                nang = ang / np.expand_dims(np.sqrt(ang[:,0]**2 + ang[:,1]**2), 1)
                orrs = np.apply_over_axes(np.mean, self.parts, [1, 2]).reshape((self._num_parts, -1))
                if E == 8:
                    orrs = orrs[...,:4] + orrs[...,4:]    
                nang = nang[:4]
                norrs = orrs / np.expand_dims(orrs.sum(axis=1), 1)
                dirs = (np.expand_dims(norrs, -1) * nang).sum(axis=1)
                self.orientations = np.asarray([math.atan2(x[1], x[0]) for x in dirs])
                II = np.argsort(self.orientations)

            II = np.argsort(scores)

            scores = scores[II]
            counts = counts[II]

            self.extra['scores'] = scores
            self.extra['counts'] = counts
            self.extra['originals'] = [self.extra['originals'][ii] for ii in II]
            
            # Now resort the parts according to this sorting
            self.orientations = self.orientations[II]
            self.parts = self.parts[II]
            #self.unspread_parts = self.unspread_parts[II]
            #self.unspread_parts_padded = self.unspread_parts_padded[II]
            self.unspread_parts = None
            self.unspread_parts_padded = None
            self.visparts = self.visparts[II]

        self._preprocess_logs()

    def _preprocess_logs(self):
        """Pre-loads log values for easy extraction of parts from edge maps"""
        self._log_parts = np.log(self.parts)
        self._log_invparts = np.log(1-self.parts)

    def _extract_edges(self, image):
        sett = self.bedges_settings().copy()
        sett['radius'] = 0
        sett['preserve_size'] = True
    
        #return np.concatenate([gv.gradients.extract(image, orientations=8), ag.features.bedges(image, **sett)], axis=2)
        #return np.concatenate([gv.gradients.extract(image, orientations=8), gv.gradients.extract(image, orientations=8, threshold=1.5)], axis=2)
        #return ag.features.bedges(image, **sett)
        if 1:
            return gv.gradients.extract(image, 
                                        orientations=8, 
                                        threshold=self.settings.get('threshold2', 0.001),
                                        eps=self.settings.get('eps', 0.001), 
                                        blur_size=self.settings.get('blur_size', 10))

    def extract_features(self, image, settings={}, dropout=None):
        sett = self.bedges_settings().copy()
        sett['radius'] = 0
        if 1:
            #unspread_edges = ag.features.bedges(image, **sett)
            #unspread_edges = gv.gradients.extract(image, orientations=8)
            unspread_edges = self._extract_edges(image) 
        else:
            # LEAVE-BEHIND: From multi-channel images
            unspread_edges = ag.features.bedges_from_image(image, **sett)
    
        # Now do spreading
        edges = ag.features.bspread(unspread_edges, spread=self.bedges_settings()['spread'], radius=self.bedges_settings()['radius'])  

        # TODO Temporary
        #sett['preserve_size'] = True
        #unspread_edges = ag.features.bedges(image, **sett)

        th = self.threshold_in_counts(self.settings['threshold'], edges.shape[-1])

        # TODO : Since we're using a hard-coded tau
        if self.settings.get('tau', 0) == 0:
            sett = self.settings.copy()
            sett.update(settings)
            psize = sett.get('subsample_size', (1, 1))
            if 0:
                feats = ag.features.extract_parts_adaptive_EXPERIMENTAL(edges, unspread_edges, 
                                                  self._log_parts,
                                                  self._log_invparts,
                                                  th,
                                                  self.settings['patch_frame'],
                                                  spread_radii=sett.get('spread_radii', (0, 0)),
                                                  subsample_size=psize,
                                                  collapse=2,
                                                  accept_threshold=15)
            elif 1:
                feats = ag.features.extract_parts(edges, unspread_edges, 
                                                  self._log_parts,
                                                  self._log_invparts,
                                                  th,
                                                  self.settings['patch_frame'],
                                                  spread_radii=sett.get('spread_radii', (0, 0)),
                                                  subsample_size=psize,
                                                  collapse=2)

            else:
                all_feats = []
                for i in xrange(10):
                    feats = ag.features.extract_parts(edges, unspread_edges, 
                                                      self._log_parts,
                                                      self._log_invparts,
                                                      th,
                                                      self.settings['patch_frame'],
                                                      spread_radii=sett.get('spread_radii', (0, 0)),
                                                      subsample_size=psize,
                                                      collapse=2,
                                                      accept_threshold=-100000)
                    all_feats.append(feats)
                all_feats = np.asarray(all_feats)

                feats = all_feats.max(axis=0)
            
            buf = tuple(image.shape[i] - feats.shape[i] * psize[i] for i in xrange(2))
            lower = (buf[0]//2, buf[1]//2)
            upper = tuple(image.shape[i] - (buf[i]-lower[i]) for i in xrange(2))

        else:
            sett = self.settings.copy()
            sett.update(settings)

            feats = self.extract_parts(edges, unspread_edges, settings=sett, dropout=dropout)

            psize = sett.get('subsample_size', (1, 1))
            feats = gv.sub.subsample(feats, psize)

            buf = tuple(image.shape[i] - feats.shape[i] * psize[i] for i in xrange(2))
            lower = (buf[0]//2, buf[1]//2)
            upper = tuple(image.shape[i] - (buf[i]-lower[i]) for i in xrange(2))

            # Now collapse the polarities
            #feats = feats.reshape((int(feats.shape[0]//2), 2) + feats.shape[1:])
            feats = feats.reshape(feats.shape[:2] + (feats.shape[2]//2, 2))
            feats = feats.max(axis=-1)

        # TODO: Experiment
        if 0:
            U = np.load('U.npy')
            new_feats = np.empty(feats.shape, dtype=np.uint8)
            for i, j in itr.product(xrange(feats.shape[0]), xrange(feats.shape[1])):
                # Transform the basis 0.572
                new_feats[i,j] = (np.fabs(np.dot(feats[i,j], U)) >= 0.5).astype(np.uint8)
            feats = new_feats

        return gv.ndfeature(feats, lower=lower, upper=upper)

    # How could it know num_edges without inputting it? 
    def threshold_in_counts(self, threshold, num_edges):
        size = self.settings['part_size']
        frame = self.settings['patch_frame']
        return int(threshold * (size[0] - 2*frame) * (size[1] - 2*frame) * num_edges)

    def extract_partprobs_from_edges(self, edges, edges_unspread):
        partprobs = ag.features.code_parts(edges, 
                                           edges_unspread,
                                           self._log_parts, self._log_invparts, 
                                           self.threshold_in_counts(self.settings['threshold'], edges.shape[-1]), self.settings['patch_frame'],
                                           max_threshold=self.threshold_in_counts(self.settings.get('max_threshold', 1.0), edges.shape[-1]))
        return partprobs

    def extract_partprobs(self, image):
        edges = ag.features.bedges(image, **self.bedges_settings())
        sett = self.bedges_settings().copy()
        sett['radius'] = 0
        edges_unspread = ag.features.bedges(image, **sett)
        return self.extract_partprobs_from_edges(edges, edges_unspread)

    def extract_parts(self, edges, edges_unspread, settings={}, dropout=None):
        #print 'strides', self.settings.get('strides', 1)
        if 'indices' in self.extra:
            feats = ag.features.code_parts_as_features_INDICES(edges, 
                                                       edges_unspread,
                                                       self._log_parts, self._log_invparts, 
                                                       self.extra['indices'],
                                                       self.threshold_in_counts(self.settings['threshold'], edges.shape[-1]), self.settings['patch_frame'], 
                                                       strides=self.settings.get('strides', 1), 
                                                       tau=self.settings.get('tau', 0.0),
                                                       max_threshold=self.threshold_in_counts(self.settings.get('max_threshold', 1.0), edges.shape[-1]))
        elif 0:
            feats = ag.features.code_parts_as_features(edges, 
                                                       edges_unspread,
                                                       self._log_parts, self._log_invparts, 
                                                       self.threshold_in_counts(self.settings['threshold'], edges.shape[-1]), self.settings['patch_frame'], 
                                                       strides=self.settings.get('strides', 1), 
                                                       tau=self.settings.get('tau', 0.0),
                                                       max_threshold=self.threshold_in_counts(self.settings.get('max_threshold', 1.0), edges.shape[-1]))
        else:
            partprobs = ag.features.code_parts(edges,
                                               edges_unspread,
                                               self._log_parts, self._log_invparts, 
                                               self.threshold_in_counts(self.settings['threshold'], edges.shape[-1]), self.settings['patch_frame'], 
                                               strides=self.settings.get('strides', 1), 
                                               max_threshold=self.threshold_in_counts(self.settings.get('max_threshold', 1.0), edges.shape[-1]))

            if dropout is not None:
                II = np.arange(self.num_features)
                np.random.shuffle(II)
                II = II[:int(dropout*self.num_features)]
                IJ = np.sort(np.concatenate([II*2, (II*2+1)]))
                yesno = np.concatenate([[0], np.bincount(IJ, minlength=self.num_parts*2)]).astype(bool)

                partprobs[:,:,yesno] = -np.inf

            # Sort out low-probability ones

            if 0:
                L = -np.ones(partprobs.shape[-1]) * np.inf
                L[0] = 0
                
                max2 = scipy.stats.scoreatpercentile(partprobs, 98, axis=-1)

                partprobs[max2 < -195] = L

               #feats = ag.features.convert_partprobs_to_feature_vector(partprobs.astype(np.float32), 2.0)
                feats = ag.features.convert_part_to_feature_vector(partprobs.argmax(axis=-1).astype(np.uint32), self.num_parts*2)
            else:
                #feats = ag.features.convert_part_to_feature_vector(partprobs.argmax(axis=-1).astype(np.uint32), self.num_parts*2)
                #feats = convert_partprobs_to_feature_vector(partprobs.astype(np.float32), 0.0)
                feats = ag.features.convert_partprobs_to_feature_vector(partprobs.astype(np.float32), self.settings.get('tau', 0))


        # Pad with background (TODO: maybe incorporate as an option to code_parts?)
        # This just makes things a lot easier, and we don't have to match for instance the
        # support which will be bigger if we don't do this.
        # TODO: If we're not using a support, this could be extremely detrimental!

        if settings.get('preserve_size'):
            # TODO: Create a function called pad_to_size that handles this better
            feats = ag.util.zeropad(feats, (self._log_parts.shape[1]//2, self._log_parts.shape[2]//2, 0))

            # TODO: This is a bit of a hack. This makes it handle even-sized parts
            if self._log_parts.shape[1] % 2 == 0:
                feats = feats[:-1]
            if self._log_parts.shape[2] % 2 == 0:
                feats = feats[:,:-1]

        sett = self.settings.copy()
        sett.update(settings)

        # Do spreading
        radii = sett.get('spread_radii', (0, 0))

        assert radii[0] == radii[1], 'Supports only axis symmetric radii spreading at this point'
        from amitgroup.features.features import array_bspread_new
        
        feats = array_bspread_new(feats, spread='box', radius=radii[0])

        cb = sett.get('crop_border')
        if cb:
            # Due to spreading, the area of influence can be greater
            # than what we're cutting off. That's why it's good to have
            # a cut_border property if you're training on real images.
            feats = feats[cb:-cb, cb:-cb]

        return feats 
    
    def __OLD_extract_parts(self, edges, settings={}, support_mask=None):
        if support_mask is not None: 
            partprobs = ag.features.code_parts_support_mask(edges, self._log_parts, self._log_invparts, 
                                               self.settings['threshold'], support_mask[2:-2,2:-2].astype(np.uint8), self.settings['patch_frame'])
        else:
            partprobs = ag.features.code_parts(edges, self._log_parts, self._log_invparts, 
                                               self.settings['threshold'], self.settings['patch_frame'], strides=1)

        tau = self.settings.get('tau', 0.0)
        #if self.settings.get('tau'):

        parts = partprobs.argmax(axis=-1)

        # Pad with background (TODO: maybe incorporate as an option to code_parts?)
        # This just makes things a lot easier, and we don't have to match for instance the
        # support which will be bigger if we don't do this.
        # TODO: If we're not using a support, this could be extremely detrimental!

        if settings.get('preserve_size'):
            parts = ag.util.zeropad(parts, (self._log_parts.shape[1]//2, self._log_parts.shape[2]//2))
            partprobs = ag.util.zeropad(partprobs, (self._log_parts.shape[1]//2, self._log_parts.shape[2]//2, 0))

            # TODO: This is a bit of a hack. This makes it handle even-sized parts
            if self._log_parts.shape[1] % 2 == 0:
                parts = parts[:-1]
                partprobs = partprobs[:-1]
            if self._log_parts.shape[2] % 2 == 0:
                parts = partprobs[:,:-1]
                partprobs = partprobs[:,:-1]

        sett = self.settings.copy()
        sett.update(settings)

        # Do spreading
        radii = sett.get('spread_radii', (0, 0))

        print tau
        spread_parts = ag.features.spread_patches_new(partprobs.astype(np.float32), radii[0], radii[1], tau)

        cb = sett.get('crop_border')
        if cb:
            # Due to spreading, the area of influence can be greater
            # than what we're cutting off. That's why it's good to have
            # a cut_border property if you're training on real images.
            spread_parts = spread_parts[cb:-cb, cb:-cb]

        return spread_parts 
        #else:
            # TODO: Maybe not this way.
            #spread_parts = ag.features.spread_parts(parts, 0, 0, self.num_parts)
            #return spread_parts 
            #return parts

    @classmethod
    def load_from_dict(cls, d):
        patch_size = d['patch_size']
        num_parts = d['num_parts']
        obj = cls(patch_size, num_parts)
        obj.parts = d['parts']
        # TODO: Experimental
        obj.unspread_parts = d['unspread_parts']
        obj.unspread_parts_padded = d['unspread_parts_padded']
        obj.visparts = d['visparts']
        obj.settings = d['settings']
        obj.orientations = d.get('orientations')
        obj._preprocess_logs()
        obj.extra = d.get('extra', {})
        return obj

    def save_to_dict(self):
        # TODO: Experimental
        #return dict(num_parts=self.num_parts, patch_size=self.patch_size, parts=self.parts, visparts=self.visparts, settings=self.settings)
        return dict(num_parts=self._num_parts, 
                    patch_size=self.patch_size, 
                    parts=self.parts, 
                    unspread_parts=self.unspread_parts, 
                    unspread_parts_padded=self.unspread_parts_padded, 
                    visparts=self.visparts, 
                    orientations=self.orientations,
                    settings=self.settings,
                    extra=self.extra)

