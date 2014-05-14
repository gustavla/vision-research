from __future__ import absolute_import
import random
import copy
import amitgroup as ag
import numpy as np
import gv
import math
import itertools as itr
from .binary_descriptor import BinaryDescriptor

# Requires package pnet

@BinaryDescriptor.register('binary-tree-parts')
class BinaryTreePartsDescriptor(BinaryDescriptor):
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
        #self.settings['strides'] = 1
    
        # TODO
        self.extra = {}

        # Or maybe just do defaults?
        # self.settings['bedges'] = {}
        self.settings['bedges'] = dict(k=5, radius=1, minimum_contrast=0.05, contrast_insensitive=True, max_edges=2)
        self.settings.update(settings)

        split_criterion = self.settings.get('split_criterion', 'IG')
        split_threshold = self.settings.get('split_threshold', 0.1)

        edge_count = [8, 4][self.settings['bedges']['contrast_insensitive']]

        import pnet
        self._net = pnet.PartsNet([
            pnet.EdgeLayer(**self.settings['bedges']),
            pnet.BinaryTreePartsLayer(self.settings.get('tree_depth', 10), 
                                      self.patch_size, 
                                      settings=dict(outer_frame=self.settings['patch_frame'], 
                                              em_seed=self.settings.get('train_em_seed', 0),
                                              threshold=self.threshold_in_counts(self.settings['threshold'],
                                                                                 edge_count), 
                                              samples_per_image=self.settings['samples_per_image'], 
                                              max_samples=1000000,
                                              train_limit=10000,
                                              min_prob=self.settings['min_probability'],
                                              #keypoint_suppress_radius=1,
                                              min_samples_per_part=50,
                                              split_criterion=split_criterion,
                                              split_entropy=split_threshold,
                                              min_information_gain=split_threshold, 
                                              )),
            pnet.PoolingLayer(shape=(9, 9), strides=(4, 4)),
        ])

    @property
    def num_parts(self):
        return self._num_parts

    @property
    def num_features(self):
        return self._num_parts

    @property
    def num_true_parts(self):
        return self._num_parts

    @property
    def subsample_size(self):
        return self.settings['subsample_size']

    def bedges_settings(self):
        return self.settings['bedges']

    def train_from_images(self, filenames):
        ag.info('Fetching training images for learning parts')

        def size_ok(im):
            return np.min(im.shape[:2]) >= 300
        def crop(im):
            return gv.img.crop(im, (300, 300))

        ims = map(crop, filter(size_ok, [gv.img.asgray(gv.img.load_image(fn)) for fn in filenames]))
        ims = np.asarray(ims)
        ag.info('Training images shape', ims.shape)
        ag.info('Training parts')
        self._net.train(ims)
        self._num_parts = self._net.layers[1].num_parts

    def extract_features(self, image, settings={}, must_preserve_size=False):
        sett = self.settings
        sett.update(settings)

        radii = sett.get('spread_radii', (0, 0))
        psize = sett.get('subsample_size', (1, 1))

        spread_shape = (radii[0] * 2 + 1, radii[1] * 2 + 1)

        self._net.layers[-1]._shape = spread_shape 
        self._net.layers[-1]._strides = psize

        feats = self._net.extract(image[np.newaxis])[0]

        buf = tuple(image.shape[i] - feats.shape[i] * psize[i] for i in xrange(2))
        lower = (buf[0]//2, buf[1]//2)
        upper = tuple(image.shape[i] - (buf[i]-lower[i]) for i in xrange(2))

        return gv.ndfeature(feats, lower=lower, upper=upper)


    # How could it know num_edges without inputting it? 
    def threshold_in_counts(self, threshold, num_edges):
        size = self.patch_size
        frame = self.settings['patch_frame']
        return int(threshold * (size[0] - 2*frame) * (size[1] - 2*frame) * num_edges)

    @classmethod
    def load_from_dict(cls, d):
        patch_size = d['patch_size']
        num_parts = d['num_parts']
        obj = cls(patch_size, num_parts)
        # TODO: Experimental
        obj.settings = d['settings']
        obj._net = d['net']
        obj.extra = d.get('extra', {})
        return obj

    def save_to_dict(self):
        # TODO: Experimental
        #return dict(num_parts=self.num_parts, patch_size=self.patch_size, parts=self.parts, visparts=self.visparts, settings=self.settings)
        return dict(num_parts=self.num_parts, 
                    patch_size=self.patch_size, 
                    settings=self.settings,
                    net=self._net,
                    extra=self.extra)

