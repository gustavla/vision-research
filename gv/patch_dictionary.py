import random
import copy
import amitgroup as ag
import numpy as np

class PatchDictionary(object):
    def __init__(self, patch_size, K, settings={}):
        self.patch_size = patch_size
        self.K = K

        self.patches = None
        self.vispatches = None

        self.settings = {}
        self.settings['patch_frame'] = 1
        self.settings['threshold'] = 4 
        self.settings['threaded'] = False 
        self.settings['samples_per_image'] = 500 
        self.settings['spread_0_dim'] = 3 
        self.settings['spread_1_dim'] = 3 

        # Or maybe just do defaults?
        # self.settings['bedges'] = {}
        self.settings['bedges'] = dict(k=5, radius=0, minimum_contrast=0.05, contrast_insensitive=False)
        for k, v in settings.items():
            self.settings[k] = v

    def _get_patches(self, filename):
        samples_per_image = self.settings['samples_per_image']
        fr = self.settings['patch_frame']
        the_patches = []
        the_originals = []
        ag.info("Extracting patches from", filename)
        #edges, img = ag.features.bedges_from_image(f, k=5, radius=1, minimum_contrast=0.05, contrast_insensitive=False, return_original=True, lastaxis=True)
        edges, img = ag.features.bedges_from_image(filename, return_original=True, **self.settings['bedges'])

        s = self.settings['bedges'].copy()
        if 'radius' in s:
            del s['radius']
        edges_nospread = ag.features.bedges_from_image(filename, radius=0, **s)

        # How many patches could we extract?
        w, h = [edges.shape[i]-self.patch_size[i]+1 for i in xrange(2)]

        # TODO: Maybe shuffle an iterator of the indices?

        for sample in xrange(samples_per_image):
            for tries in xrange(20):
                x, y = random.randint(0, w-1), random.randint(0, h-1)
                selection = [slice(x, x+self.patch_size[0]), slice(y, y+self.patch_size[1])]
                # Return grayscale patch and edges patch
                edgepatch = edges[selection]
                edgepatch_nospread = edges_nospread[selection]
                num = edgepatch_nospread[fr:-fr,fr:-fr].sum()
                if num >= self.settings['threshold']: 
                    the_patches.append(edgepatch_nospread)
        
                    # The following is only for clearer visualization of the 
                    # patches. However, normalizing like this might be misleading
                    # in other ways.
                    vispatch = img[selection]
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
            if self.settings['threaded']:
                from multiprocessing import Pool
                p = Pool(8) # Should not be hardcoded
                mapfunc = p.map
            else:
                mapfunc = map

        ret = map(self._get_patches, filenames)

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

        mixture = ag.stats.BernoulliMixture(self.K, raw_patches, init_seed=0)
        # Also store these in "settings"
        mixture.run_EM(1e-8, min_probability=0.05)
        ag.info("Done.")
        
        # Store the stuff in the instance
        self.patches = mixture.templates
        self.vispatches = mixture.remix(raw_originals)

        self._preload_logs()

    def _preload_logs(self):
        """Pre-loads log values for easy extraction of parts from edge maps"""
        self._log_parts = np.log(self.patches)
        self._log_invparts = np.log(1-self.patches)

    def extract_parts(self, edges):
        arr = []
        
        s0, s1 = self.settings['spread_0_dim'], self.settings['spread_1_dim']

        partprobs = ag.features.code_parts(edges, self._log_parts, self._log_invparts, 
                                           self.settings['threshold'], self.settings['patch_frame'])
        parts = partprobs.argmax(axis=0)
        
        spread_parts = ag.features.spread_patches(parts, s0, s1, self.K)
        arr.append(spread_parts)  

        return arr 
           

    @classmethod
    def load(cls, path):
        data = np.load(path)
        patch_size = data['patch_size'].flat[0]
        K = data['K'].flat[0]
        obj = cls.__class__(patch_size, K)
        obj.patches = data['patches']
        obj.vispatches = data['vispatches']
        obj.settings = data['settings'].flat[0]
        obj._preload_logs()
        return
        
    def save(self, path):
        if self.patches is None:
            raise Exception("PartsDictionary not trained yet")

        np.savez(path, K=self.K, patch_size=self.patch_size, patches=self.patches, vispatches=self.vispatches, settings=self.settings)

