
import numpy as np
import amitgroup as ag
import random

class PatchModel:
    def __init__(self, patch_size, K):
        self.patches = None
        self.vispatches = None
        self.patch_size = patch_size
        self.big_patch_frame = 1
        self.K = K

    def _gen_patches(self, img, edges, patch_size):
        for x in xrange(img.shape[0]-patch_size[0]+1):
            for y in xrange(img.shape[1]-patch_size[1]+1):
                selection = [slice(x, x+patch_size[0]), slice(y, y+patch_size[1])]
                # Return grayscale patch and edges patch
                yield img[selection], edges[selection]

    def train_patches(self, data):
        pass

    def train_from_image_files(self, filenames):
        self.train_patches_from_image_files(filenames)

        # Now, train the model
        self.train_model(filenames)

    def train_patches_from_image_files(self, filenames, samples_per_image=1000):
        random.seed(0)
        filenames_copy = filenames[:]
        random.shuffle(filenames_copy)

        raw_patches = []
        raw_originals = [] 

        fr = self.big_patch_frame
    
        for file_i, f in enumerate(filenames_copy):
            ag.info(file_i, "File", f)
            edges, img = ag.features.bedges_from_image(f, k=5, radius=1, minimum_contrast=0.05, contrast_insensitive=True, return_original=True)
            # Make grayscale
            imggray = img[...,:3].mean(axis=2)
            for impatch, edgepatch in self._gen_patches(imggray, edges, self.patch_size):
                if edgepatch[fr:-fr,fr:-fr].sum() >= 20:
                    raw_originals.append(impatch)
                    raw_patches.append(edgepatch)  

        raw_patches = np.asarray(raw_patches)
        raw_originals = np.asarray(raw_originals)

        mixture = ag.stats.BernoulliMixture(self.K, raw_patches, init_seed=0)
        mixture.run_EM(1e-4, min_probability=0.01, debug_plot=False)

        # Store the stuff in the instance
        self.patches = mixture.templates
        self.vispatches = mixture.remix(raw_originals)
        self.vispatches /= self.vispatches.max()

        #self.info = {
        #    'K': self.K,
        #    'patch_size': self.patch_size,
        #    'big_patch_frame': self.big_patch_frame,
        #}

        return self.patches

    def features_from_image(self, image):
        edges, img = ag.features.bedges_from_image(image, k=5, radius=1, minimum_contrast=0.05, contrast_insensitive=True, return_original=True)
        return edges, img
        
    def train_model(self, filenames):
        data = []
        for f in filenames[:50]:
            ag.info("Preparing", f)
            #edges, img = ag.features.bedges_from_image(f, k=5, radius=1, minimum_contrast=0.05, contrast_insensitive=True, return_original=True)
            edges, img = self.features_from_image(f)
            print "edges.shape =", edges.shape
            data.append(edges)

        data = np.asarray(data)
        print "data.shape =", data.shape
         

    def train(self, data):
        self.train_patches(data) 

    def save(self, output_file):
        np.savez(output_file, patches=self.patches, originals=self.vispatches, info=self.info)

    def load(self, fn):
        pass


def _gen_patches(self, img, edges, patch_size):
    for x in xrange(img.shape[0]-patch_size[0]+1):
        for y in xrange(img.shape[1]-patch_size[1]+1):
            selection = [slice(x, x+patch_size[0]), slice(y, y+patch_size[1])]
            # Return grayscale patch and edges patch
            yield img[selection], edges[selection]


def get_patches(args):
    f, patch_size, samples_per_image, fr = args
    the_patches = []
    the_originals = []
    ag.info("File", f)
    edges, img = ag.features.bedges_from_image(f, k=5, radius=1, minimum_contrast=0.05, contrast_insensitive=False, return_original=True)
    edges_nospread = ag.features.bedges_from_image(f, k=5, radius=0, minimum_contrast=0.05, contrast_insensitive=False)

    # How many patches could we extract?
    w, h = [edges.shape[i]-patch_size[i]+1 for i in xrange(2)]
    #if samples_per_image is None:
        ## Get all of them
        #indices = range(w * h)
    #else:
        ## Get samples from this
        #indices = random.sample(xrange(w * h), samples_per_image) 

    #indices = range(w * h)
    #random.shuffle(indices)

    #positions = map(lambda index: (index%w, index/w), indices)

    #ag.plot.images([img])

    #for x, y in positions:
    for sample in xrange(samples_per_image):
        for tries in xrange(20):
            x, y = random.randint(0, w-1), random.randint(0, h-1)
            selection = [slice(x, x+patch_size[0]), slice(y, y+patch_size[1])]
            # Return grayscale patch and edges patch
            edgepatch = edges[selection]
            edgepatch_nospread = edges_nospread[selection]
            num = edgepatch[fr:-fr,fr:-fr].sum()
            #num_edges.append(num)
            if num >= 4: 
                the_patches.append(edgepatch_nospread)
    
                vispatch = img[selection]
                vispatch = vispatch[...,:3].mean(axis=vispatch.ndim-1)

                span = vispatch.min(), vispatch.max() 
                if span[1] - span[0] > 0:
                    vispatch = (vispatch-span[0])/(span[1]-span[0])
                the_originals.append(vispatch)
                break

    return the_patches, the_originals

from multiprocessing import Pool

def random_patches(filenames, patch_size, seed=0, samples_per_image=None):
    random.seed(seed)
    filenames_copy = filenames[:]
    random.shuffle(filenames_copy)

    raw_patches = []
    raw_originals = [] 

    fr = 1 
    p = Pool(8)
    ret = p.map(get_patches, [(f, patch_size, samples_per_image, fr) for f in filenames_copy])

    for patches, originals in ret:
        raw_patches.extend(patches)
        raw_originals.extend(originals) 

    raw_patches = np.asarray(raw_patches)
    raw_originals = np.asarray(raw_originals)
    
    return raw_patches, raw_originals
