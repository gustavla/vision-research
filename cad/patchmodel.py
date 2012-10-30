
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
        self.side = 

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

    def train_patches_from_image_files(self, filenames):
        random.seed(0)
        filenames_copy = filenames[:]
        random.shuffle(filenames_copy)

        raw_patches = []
        raw_originals = [] 

        fr = self.big_patch_frame
    
        N = 10

        for f in filenames_copy[:N]:
            ag.info("File", f)
            edges, img = ag.features.bedges_from_image(f, k=5, radius=1, minimum_contrast=0.05, contrast_insensitive=True, return_original=True, lastaxis=True)
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

        self.info = {
            'K': self.K,
            'patch_size': self.patch_size,
            'big_patch_frame': self.big_patch_frame,
        }

    def features_from_image(self, image):
        edges, img = ag.features.bedges_from_image(image, k=5, radius=1, minimum_contrast=0.05, contrast_insensitive=True, return_original=True, lastaxis=True)
        return edges, img
        
    def train_model(self, filenames):
        data = []
        for f in filenames[:50]:
            ag.info("Preparing", f)
            #edges, img = ag.features.bedges_from_image(f, k=5, radius=1, minimum_contrast=0.05, contrast_insensitive=True, return_original=True, lastaxis=True)
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

