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

def _threshold_in_counts(settings, num_edges):
    threshold = settings['threshold']
    size = settings['part_size']
    frame = settings['patch_frame']
    return max(1, int(threshold * (size[0] - 2*frame) * (size[1] - 2*frame) * num_edges))

def _extract_many_edges(bedges_settings, settings, images, must_preserve_size=False):
    """Extract edges of many images (must be the same size)"""
    sett = bedges_settings.copy()
    sett['radius'] = 0
    sett['preserve_size'] = False or must_preserve_size

    edge_type = settings.get('edge_type', 'yali')
    if edge_type == 'yali':
        return ag.features.bedges(images, **sett)
    elif edge_type == 'new':
        return np.asarray([gv.gradients.extract(image, 
                                    orientations=8, 
                                    threshold=settings.get('threshold2', 0.001),
                                    eps=settings.get('eps', 0.001), 
                                    blur_size=settings.get('blur_size', 10)) for image in images])
    else:
        raise RuntimeError("No such edge type")

def _extract_edges(bedges_settings, settings, image, must_preserve_size=False):
    return _extract_many_edges(bedges_settings, settings, image[np.newaxis], must_preserve_size=must_preserve_size)[0]

def _get_patches(bedges_settings, settings, filename):
    samples_per_image = settings['samples_per_image']
    fr = settings['patch_frame']
    the_patches = []
    the_originals = []
    ag.info("Extracting patches from", filename)
    #edges, img = ag.features.bedges_from_image(f, k=5, radius=1, minimum_contrast=0.05, contrast_insensitive=False, return_original=True, lastaxis=True)
    setts = settings['bedges'].copy()
    radius = setts['radius']
    setts2 = setts.copy()
    setts['radius'] = 0
    ps = settings['part_size']


    ORI = settings.get('orientations', 1)
    POL = settings.get('polarities', 1)
    assert POL in (1, 2), "Polarity must be 1 or 2"
    #assert ORI%2 == 0, "Orientations must be even, so that opposite can be collapsed"

    # LEAVE-BEHIND

    # Fetch all rotations of this image
    img = gv.img.load_image(filename)
    img = gv.img.asgray(img)

    from skimage import transform

    size = img.shape[:2]
    # Make it square, to accommodate all types of rotations
    new_size = int(np.max(size) * np.sqrt(2))
    img_padded = ag.util.zeropad_to_shape(img, (new_size, new_size))
    pad = [(new_size-size[i])//2 for i in xrange(2)]

    angles = np.arange(0, 360, 360/ORI)
    radians = angles*np.pi/180
    all_img = np.asarray([transform.rotate(img_padded, angle, resize=False) for angle in angles])
    # Add inverted polarity too
    if POL == 2:
        all_img = np.concatenate([all_img, 1-all_img])


    # Set up matrices that will translate a position in the canonical image to
    # the rotated iamges. This way, we're not rotating each patch on demand, which
    # will end up slower.
    matrices = [gv.matrix.translation(new_size/2, new_size/2) * gv.matrix.rotation(a) * gv.matrix.translation(-new_size/2, -new_size/2) for a in radians]

    # Add matrices for the polarity flips too, if applicable
    matrices *= POL 

    #inv_img = 1 - img
    all_unspread_edges = _extract_many_edges(bedges_settings, settings, all_img, must_preserve_size=True)

    #unspread_edges_padded = ag.util.zeropad(unspread_edges, (radius, radius, 0))
    #inv_unspread_edges_padded = ag.util.zeropad(inv_unspread_edges, (radius, radius, 0))

    # Spread the edges
    all_edges = ag.features.bspread(all_unspread_edges, spread=setts['spread'], radius=radius)
    #inv_edges = ag.features.bspread(inv_unspread_edges, spread=setts['spread'], radius=radius)

    #setts2['minimum_contrast'] *= 2
    #edges2 = ag.features.bedges(img, **setts2)
    #inv_edges2 = ag.features.bedges(inv_img, **setts2)
    
    #s = self.settings['bedges'].copy()
    #if 'radius' in s:
    #    del s['radius']
    #edges_nospread = ag.features.bedges_from_image(filename, radius=0, **s)

    # How many patches could we extract?

    # This avoids hitting the outside of patches, even after rotating.
    # The 15 here is fairly arbitrary
    avoid_edge = int(15 + np.max(ps)*np.sqrt(2))

    # This step assumes that the img -> edge process does not down-sample any

    # TODO: Maybe shuffle an iterator of the indices?

    # These indices represent the center of patches
    indices = list(itr.product(xrange(pad[0]+avoid_edge, pad[0]+img.shape[0]-avoid_edge), xrange(pad[1]+avoid_edge, pad[1]+img.shape[1]-avoid_edge)))
    random.shuffle(indices)
    i_iter = iter(indices)

    minus_ps = [-ps[i]//2 for i in xrange(2)]
    plus_ps = [minus_ps[i] + ps[i] for i in xrange(2)]

    E = all_edges.shape[-1]
    th = _threshold_in_counts(settings, E)

    rs = np.random.RandomState(0)

    for sample in xrange(samples_per_image):
        for tries in xrange(100):
            #selection = [slice(x, x+self.patch_size[0]), slice(y, y+self.patch_size[1])]
            #x, y = random.randint(0, w-1), random.randint(0, h-1)
            x, y = i_iter.next()

            selection0 = [0, slice(x+minus_ps[0], x+plus_ps[0]), slice(y+minus_ps[1], y+plus_ps[1])]

            # Return grayscale patch and edges patch
            unspread_edgepatch = all_unspread_edges[selection0]
            edgepatch = all_edges[selection0]
            #inv_edgepatch = inv_edges[selection]

            #amppatch = amps[selection]
            #edgepatch2 = edges2[selection]
            #inv_edgepatch2 = inv_edges2[selection]
            #edgepatch_nospread = edges_nospread[selection]

            # The inv edges could be incorproated here, but it shouldn't be that different.
            if fr == 0:
                tot = unspread_edgepatch.sum()
            else:
                tot = unspread_edgepatch[fr:-fr,fr:-fr].sum()

            #if self.settings['threshold'] <= avg <= self.settings.get('max_threshold', np.inf): 
            #print(th, tot)
            #print 'th', th
            if th <= tot:
                XY = np.matrix([x, y, 1]).T
                # Now, let's explore all orientations

                patch = np.zeros((ORI * POL,) + ps + (E,))
                vispatch = np.zeros((ORI * POL,) + ps)

                for ori in xrange(ORI * POL):
                    p = matrices[ori] * XY
                    ip = [int(round(p[i])) for i in xrange(2)]

                    selection = [ori, slice(ip[0]+minus_ps[0], ip[0]+plus_ps[0]), slice(ip[1]+minus_ps[1], ip[1]+plus_ps[1])]

                    patch[ori] = all_edges[selection]


                    orig = all_img[selection]
                    span = orig.min(), orig.max() 
                    if span[1] - span[0] > 0:
                        orig = (orig-span[0])/(span[1]-span[0])

                    vispatch[ori] = orig 

                # Randomly rotate this patch, so that we don't bias 
                # the unrotated (and possibly unblurred) image

                shift = rs.randint(ORI)

                patch[:ORI] = np.roll(patch[:ORI], shift, axis=0)
                vispatch[:ORI] = np.roll(vispatch[:ORI], shift, axis=0)

                if POL == 2:
                    patch[ORI:] = np.roll(patch[ORI:], shift, axis=0)
                    vispatch[ORI:] = np.roll(vispatch[ORI:], shift, axis=0)

                the_patches.append(patch)
                the_originals.append(vispatch)

                break

            if tries == 99:
                print "100 tries!"

    return the_patches, the_originals 

#def ok(amp, patch_frame, threshold):
    #amp_inner = amp[patch_frame:-patch_frame,patch_frame:-patch_frame]
    #return amp_inner.mean() > threshold 

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

@BinaryDescriptor.register('oriented-parts')
class OrientedPartsDescriptor(BinaryDescriptor):
    def __init__(self, patch_size, num_parts, settings={}):
        self.patch_size = patch_size
        self._num_parts = num_parts 
        self._num_true_parts = None

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
    def num_true_parts(self):
        return self._num_true_parts

    @property
    def subsample_size(self):
        return self.settings['subsample_size']

    def random_patches_from_images(self, filenames):
        raw_patches = []
        raw_unspread_patches = []
        raw_unspread_patches_padded = []
        raw_originals = [] 

        #ret = mapfunc(_get_patches, filenames)
        args = [(self.bedges_settings(), self.settings, filename) for filename in filenames]
        ret = gv.parallel.starmap_unordered(_get_patches, args)

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

        if len(raw_patches) == 0:
            raise Exception("No patches found, maybe your thresholds are too strict?")
        # Also store these in "settings"

        mixtures = []
        llhs = []
        from gv.latent_bernoulli_mm import LatentBernoulliMM

        ORI = self.settings.get('orientations', 1)
        POL = self.settings.get('polarities', 1)
        #P = self.settings['orientations'] * self.settings['polarities'] 
        P = ORI * POL

        def cycles(X):
            return np.asarray([np.concatenate([X[i:], X[:i]]) for i in xrange(len(X))])

        # Arrange permutations (probably a roundabout way of doing it)
        RR = np.arange(ORI)
        PP = np.arange(POL)
        II = [list(itr.product(PPi, RRi)) for PPi in cycles(PP) for RRi in cycles(RR)]
        lookup = dict(zip(itr.product(PP, RR), itr.count()))
        permutations = np.asarray([[lookup[ii] for ii in rows] for rows in II])
        n_init = self.settings.get('n_init', 1)
        n_iter = self.settings.get('n_iter', 10)
        seed = self.settings.get('seed', 0)
        
        mixture = LatentBernoulliMM(n_components=self._num_parts, permutations=permutations, n_init=n_init, n_iter=n_iter, random_state=seed, min_probability=self.settings['min_probability'])
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

        self.extra['means'] = scipy.ndimage.zoom(means, P, order=0)
        self.extra['stds'] = scipy.ndimage.zoom(sigmas, P, order=0)


        means2 = []
        sigmas2 = []
        
        II = comps[:,0]

        for f in xrange(self.num_features):
            X = llhs[II == f]
            if len(X) > 0:
                mu = X.mean()
                sigma = X.std()
            else:
                # This one will guaranteed be sorted out anyway
                mu = 0.0 
                sigma = 1.0

            means2.append(mu)
            sigmas2.append(sigma)

        #self.extra['means_emp'] = np.asarray(means2)
        #self.extra['stds_emp'] = np.asarray(sigmas2)

        self.extra['means_emp'] = scipy.ndimage.zoom(means2, P, order=0)
        self.extra['stds_emp'] = scipy.ndimage.zoom(sigmas2, P, order=0)

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

        self.parts = mixture.means_.reshape((mixture.n_components, P) + raw_patches.shape[2:])
        self.orientations = None 
        #self.unspread_parts = self.unspread_parts[II]
        #self.unspread_parts_padded = self.unspread_parts_padded[II]
        self.unspread_parts = None
        self.unspread_parts_padded = None
        #self.visparts = None#self.visparts[II]

        firsts = np.zeros(self.parts.shape[0], dtype=int)
        for f in xrange(self.parts.shape[0]): 
            avgs = np.apply_over_axes(np.mean, self.parts[f,:ORI,...,0], [1, 2]).squeeze()
            firsts[f] = np.argmax(avgs)

            self.parts[f,:ORI] = np.roll(self.parts[f,:ORI], -firsts[f], axis=0)
            if POL == 2:
                self.parts[f,ORI:] = np.roll(self.parts[f,ORI:], -firsts[f], axis=0)

        # Rotate the parts, so that the first orientation is dominating in the first edge feature.

        #visparts = mixture.remix(raw_originals)
        #aff = np.asarray(self.affinities)
        #self.visparts = np.asarray([np.average(data, axis=0, weights=aff[:,m]) for m in xrange(self.num_mix)])
        self.visparts = np.zeros((mixture.n_components,) + raw_originals.shape[2:])
        counts = np.zeros(mixture.n_components)
        for n in xrange(raw_originals.shape[0]):
            f, polarity = comps[n]# np.unravel_index(np.argmax(mixture.q[n]), mixture.q.shape[1:])
            self.visparts[f] += raw_originals[n,(polarity+firsts[f])%P]
            counts[f] += 1
        self.visparts /= counts[:,np.newaxis,np.newaxis].clip(min=1)

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


        self.parts = self.parts.reshape((self.parts.shape[0] * P,) + self.parts.shape[2:])

        #order_single = np.argsort(means / sigmas)
        order_single = np.argsort(means)
        new_order_single = []
        for f in order_single: 
            part = self.parts[P*f] 
            sh = part.shape
            p = part.reshape((sh[0]*sh[1], sh[2]))
            
            pec = p.mean(axis=0)

            N = np.sum(p * np.log(p/pec) + (1-p)*np.log((1-p)/(1-pec)))
            D = np.sqrt(np.sum(np.log(p/pec * (1-pec)/(1-p))**2 * p * (1-p)))

            ok = (N/D > 1) and counts[f] > 100
              
            if ok: 
                new_order_single.append(f)

        order_single = np.asarray(new_order_single)

        assert len(order_single) > 0, "No edges kept! Something probably went wrong"

        #self._num_parts = len(order_single)
        self._num_parts = len(order_single) * ORI 
        self._num_true_parts = len(order_single)
        print 'num_parts', self._num_parts
        print 'num_true_parts', self._num_true_parts

        order = np.zeros(len(order_single) * P, dtype=int) 
        for f in xrange(len(order_single)):
            for p in xrange(P):
                order[P*f+p] = order_single[f]*P+p

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

            II0 = np.where(comps[:,0] == i)[0]
            II1 = comps[II0,1]

            ims = np.asarray([raw_originals[tuple(ij)] for ij in itr.izip(II0, II1)])
            #ims = raw_originals[comps[:,0] == i,comps[:,1]].copy()
            originals.append(ims[:50])
        self.extra['originals'] = originals


        self._preprocess_logs()

    def _preprocess_logs(self):
        """Pre-loads log values for easy extraction of parts from edge maps"""
        self._log_parts = np.log(self.parts)
        self._log_invparts = np.log(1-self.parts)

    def extract_features(self, image, settings={}, must_preserve_size=False, dropout=None):
        sett = self.bedges_settings().copy()
        sett['radius'] = 0
        if 1:
            #unspread_edges = ag.features.bedges(image, **sett)
            #unspread_edges = gv.gradients.extract(image, orientations=8)
            unspread_edges = _extract_edges(self.bedges_settings(), self.settings, image, must_preserve_size=must_preserve_size) 
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

            ORI = self.settings.get('orientations', 1)
            POL = self.settings.get('polarities', 1)
            P = ORI * POL 
            H = P // 2

            if POL == 2:
                part_to_feature = np.zeros(self.parts.shape[0], dtype=np.int64)
                for f in xrange(part_to_feature.shape[0]):
                    thepart = f // P
                    ori = f % H
                    v = thepart * H + ori

                    part_to_feature[f] = v 
            else:
                part_to_feature = np.arange(self.parts.shape[0], dtype=np.int64)

            # Rotation spreading?
            rotspread = sett.get('rotation_spreading_radius', 0)
            if rotspread == 0:
                between_feature_spreading = None
            else:
                between_feature_spreading = np.zeros((self.num_parts, rotspread*2 + 1), dtype=np.int32)

                for f in xrange(self.num_parts):
                    thepart = f // ORI
                    ori = f % ORI 
                    for i in xrange(rotspread*2 + 1):
                        between_feature_spreading[f,i] = thepart * ORI + (ori - rotspread + i) % ORI

            feats = ag.features.extract_parts(edges, unspread_edges,
                                              self._log_parts,
                                              self._log_invparts,
                                              th,
                                              self.settings['patch_frame'],
                                              spread_radii=sett.get('spread_radii', (0, 0)),
                                              subsample_size=psize,
                                              part_to_feature=part_to_feature,
                                              stride=self.settings.get('part_coding_stride', 1),
                                              between_feature_spreading=between_feature_spreading)

            
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
            Q = np.load('Q.npy')
            new_feats_shape = feats.shape
            new_feats = np.empty(new_feats_shape, dtype=np.uint8)
            for i, j in itr.product(xrange(feats.shape[0]), xrange(feats.shape[1])):
                # Transform the basis 0.572
                #new_feats[i,j] = np.dot(Q[:,-ARTS:].T, feats[i,j])
                new_feats[i,j] = (np.fabs(np.dot(Q.T, feats[i,j].astype(float))) > 15)
            feats = new_feats

        return gv.ndfeature(feats, lower=lower, upper=upper)

    # How could it know num_edges without inputting it? 
    def threshold_in_counts(self, threshold, num_edges):
        size = self.settings['part_size']
        frame = self.settings['patch_frame']
        return max(1, int(threshold * (size[0] - 2*frame) * (size[1] - 2*frame) * num_edges))

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
    
    @classmethod
    def load_from_dict(cls, d):
        patch_size = d['patch_size']
        num_parts = d['num_parts']
        obj = cls(patch_size, num_parts)
        obj._num_true_parts = d['num_true_parts']
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
                    num_true_parts=self._num_true_parts,
                    patch_size=self.patch_size, 
                    parts=self.parts, 
                    unspread_parts=self.unspread_parts, 
                    unspread_parts_padded=self.unspread_parts_padded, 
                    visparts=self.visparts, 
                    orientations=self.orientations,
                    settings=self.settings,
                    extra=self.extra)

