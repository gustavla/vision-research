from __future__ import absolute_import
import random
import copy
import amitgroup as ag
import numpy as np
import gv
from .binary_descriptor import BinaryDescriptor

@BinaryDescriptor.register('parts')
class PartsDescriptor(BinaryDescriptor):
    def __init__(self, patch_size, num_parts, settings={}):
        self.patch_size = patch_size
        self.num_parts = num_parts 

        self.parts = None
        self.visparts = None

        self.settings = {}
        self.settings['patch_frame'] = 1
        self.settings['threshold'] = 4 
        self.settings['threaded'] = False 
        self.settings['samples_per_image'] = 500 
        self.settings['min_probability'] = 0.005

        # Or maybe just do defaults?
        # self.settings['bedges'] = {}
        self.settings['bedges'] = dict(k=5, radius=1, minimum_contrast=0.05, contrast_insensitive=True, max_edges=2)
        self.settings.update(settings)

    def _get_patches(self, filename):
        samples_per_image = self.settings['samples_per_image']
        fr = self.settings['patch_frame']
        the_patches = []
        the_originals = []
        ag.info("Extracting patches from", filename)
        #edges, img = ag.features.bedges_from_image(f, k=5, radius=1, minimum_contrast=0.05, contrast_insensitive=False, return_original=True, lastaxis=True)

        # LEAVE-BEHIND
        if 1:
            img = gv.img.load_image(filename)
            img = gv.img.asgray(img)
            edges = ag.features.bedges(img, **self.settings['bedges'])
        else:
            edges, img = ag.features.bedges_from_image(filename, return_original=True, **self.settings['bedges'])

        #s = self.settings['bedges'].copy()
        #if 'radius' in s:
        #    del s['radius']
        #edges_nospread = ag.features.bedges_from_image(filename, radius=0, **s)

        # How many patches could we extract?
        w, h = [edges.shape[i]-self.patch_size[i]+1 for i in xrange(2)]

        # TODO: Maybe shuffle an iterator of the indices?

        for sample in xrange(samples_per_image):
            for tries in xrange(20):
                x, y = random.randint(0, w-1), random.randint(0, h-1)
                selection = [slice(x, x+self.patch_size[0]), slice(y, y+self.patch_size[1])]
                # Return grayscale patch and edges patch
                edgepatch = edges[selection]
                #edgepatch_nospread = edges_nospread[selection]
                if fr == 0:
                    num = edgepatch.sum()
                else:
                    num = edgepatch[fr:-fr,fr:-fr].sum()
                if num >= self.settings['threshold']: 
                    the_patches.append(edgepatch)
                    #the_patches.append(edgepatch_nospread)
        
                    # The following is only for clearer visualization of the 
                    # patches. However, normalizing like this might be misleading
                    # in other ways.
                    vispatch = img[selection]
                    if 1:
                        pass
                    else:
                        vispatch = vispatch[...,:3].mean(axis=vispatch.ndim-1)
                    span = vispatch.min(), vispatch.max() 
                    if span[1] - span[0] > 0:
                        vispatch = (vispatch-span[0])/(span[1]-span[0])
                    the_originals.append(vispatch)
                    break

        return the_patches, the_originals

    def random_patches_from_images(self, filenames):
        raw_patches = []
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
        if len(raw_patches) == 0:
            raise Exception("No patches found, maybe your thresholds are too strict?")
        mixture = ag.stats.BernoulliMixture(self.num_parts, raw_patches, init_seed=0)
        # Also store these in "settings"
        mixture.run_EM(1e-8, min_probability=self.settings['min_probability'])
        ag.info("Done.")

        # Reject weak parts
        scores = np.empty(self.num_parts) 
        for i in xrange(self.num_parts):
            part = mixture.templates[i]
            sh = part.shape
            p = part.reshape((sh[0]*sh[1], sh[2]))
            
            pec = p.mean(axis=0)
        
            N = np.sum(p * np.log(p/pec) + (1-p)*np.log((1-p)/(1-pec)))
            D = np.sqrt(np.sum(np.log(p/(1-p))**2 * p * (1-p)))

            scores[i] = N/D 

        # Only keep with a certain score
        visparts = mixture.remix(raw_originals)
        
        self.parts = mixture.templates[scores > 1]
        self.visparts = visparts[scores > 1]
        self.num_parts = self.parts.shape[0]
        
        # Update num_parts
        
        # Store the stuff in the instance
        #self.parts = mixture.templates
        #self.visparts = mixture.remix(raw_originals)

        self._preprocess_logs()

    def _preprocess_logs(self):
        """Pre-loads log values for easy extraction of parts from edge maps"""
        self._log_parts = np.log(self.parts)
        self._log_invparts = np.log(1-self.parts)

    def extract_features(self, image, settings={}):
        if 1:
            edges = ag.features.bedges(image, **self.bedges_settings())
        else:
            # LEAVE-BEHIND: From multi-channel images
            edges = ag.features.bedges_from_image(image, **self.bedges_settings()) 
        return self.extract_parts(edges, settings=settings)
    
    def extract_parts(self, edges, settings={}):
        partprobs = ag.features.code_parts(edges, self._log_parts, self._log_invparts, 
                                           self.settings['threshold'], self.settings['patch_frame'])

        #tau = self.settings.get('tau')
        #if self.settings.get('tau'):
        parts = partprobs.argmax(axis=-1)

        # Pad with background (TODO: maybe incorporate as an option to code_parts?)
        # This just makes things a lot easier, and we don't have to match for instance the
        # support which will be bigger if we don't do this.
        # TODO: If we're not using a support, this could be extremely detrimental!

        if settings.get('preserve_size'):
            parts = ag.util.zeropad(parts, (self._log_parts.shape[1]//2, self._log_parts.shape[2]//2))

            # TODO: This is a bit of a hack. This makes it handle even-sized parts
            if self._log_parts.shape[1] % 2 == 0:
                parts = parts[:-1]
            if self._log_parts.shape[2] % 2 == 0:
                parts = parts[:,:-1]
        
        sett = self.settings.copy()
        sett.update(settings)

        # Do spreading
        radii = sett['spread_radii']
        #radii = (0, 0)
        #if max(radii) > 0:
        spread_parts = ag.features.spread_patches(parts, radii[0], radii[1], self.num_parts)
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
        obj.visparts = d['visparts']
        obj.settings = d['settings']
        obj._preprocess_logs()
        return obj

    def save_to_dict(self):
        return dict(num_parts=self.num_parts, patch_size=self.patch_size, parts=self.parts, visparts=self.visparts, settings=self.settings)

