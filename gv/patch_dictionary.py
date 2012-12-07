import random
import copy
import amitgroup as ag
import numpy as np
from saveable import Saveable

# TODO: Move
def max_pooling(data, size):
    steps = tuple([data.shape[i]//size[i] for i in xrange(2)])
    if data.ndim == 3:
        output = np.zeros(steps + (data.shape[-1],))
    else:
        output = np.zeros(steps)

    print 'output', output.shape

    for i in xrange(steps[0]):
        for j in xrange(steps[1]):
            if data.ndim == 3: 
                output[i,j] = data[i*size[0]:(i+1)*size[0], j*size[1]:(j+1)*size[1]].max(axis=0).max(axis=0)
                #output[i,j] = data[i*size[0]:(i+1)*size[0], j*size[1]:(j+1)*size[1]].mean(axis=0).mean(axis=0)
                output[i,j] = data[i*size[0],j*size[1]]
            else:
                ##output[i,j] = data[i*size[0]:(i+1)*size[0], j*size[1]:(j+1)*size[1]].max()
                #output[i,j] = data[i*size[0]:(i+1)*size[0], j*size[1]:(j+1)*size[1]].mean()
                output[i,j] = data[i*size[0],j*size[1]]
    return output

class PatchDictionary(Saveable):
    def __init__(self, patch_size, num_patches, settings={}):
        self.patch_size = patch_size
        self.num_patches = num_patches 

        self.patches = None
        self.vispatches = None

        self.settings = {}
        self.settings['patch_frame'] = 1
        self.settings['threshold'] = 4 
        self.settings['threaded'] = False 
        self.settings['samples_per_image'] = 500 
        self.settings['spread_0_dim'] = 3 
        self.settings['spread_1_dim'] = 3 
        self.settings['pooling_size'] = (8, 8)

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

        mixture = ag.stats.BernoulliMixture(self.num_patches, raw_patches, init_seed=0)
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

    def extract_parts_from_image(self, image, spread=True, return_original=False):
        if return_original:
            edges, img = ag.features.bedges_from_image(image, return_original=True, **self.bedges_settings()) 
            return self.extract_parts(edges, spread), img       
        else: 
            edges = ag.features.bedges_from_image(image, **self.bedges_settings()) 
            return self.extract_parts(edges, spread)
    
    def extract_parts(self, edges, spread=True):
        s0, s1 = self.settings['spread_0_dim'], self.settings['spread_1_dim']
        partprobs = ag.features.code_parts(edges, self._log_parts, self._log_invparts, 
                                           self.settings['threshold'], self.settings['patch_frame'])
        parts = partprobs.argmax(axis=-1)
        
        if spread:
            spread_parts = ag.features.spread_patches(parts, s0, s1, self.num_patches)
            return spread_parts 
        else:
            # TODO: Maybe not this way.
            #spread_parts = ag.features.spread_patches(parts, 0, 0, self.num_patches)
            #return spread_parts 
            return parts

    def max_pooling(self, parts):
        return max_pooling(parts, self.settings['pooling_size'])

    def extract_pooled_parts(self, edges):
        spread_parts = self.extract_parts(edges)
        return max_pooling(spread_parts, self.settings['pooling_size'])

    @classmethod
    def load_from_dict(cls, d):
        patch_size = d['patch_size']
        num_patches = d['num_patches']
        obj = cls(patch_size, num_patches)
        obj.patches = d['patches']
        obj.vispatches = d['vispatches']
        obj.settings = d['settings']
        obj._preload_logs()
        return obj

    def save_to_dict(self):
        return dict(num_patches=self.num_patches, patch_size=self.patch_size, patches=self.patches, vispatches=self.vispatches, settings=self.settings)

