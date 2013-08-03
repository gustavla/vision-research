from __future__ import absolute_import
from .detector import *
from .real_detector import RealDetector
from . import img
from . import bb
from . import sub 
from . import rescalc
from . import datasets
# Datasets
from . import voc
from . import uiuc
from . import custom
from .beta_mixture import BetaMixture, binary_search # Temporarily exposed

from .ndfeature import ndfeature

from .binary_descriptor import *
from . import edge_descriptor
from . import parts_descriptor
from . import binary_hog_descriptor

from .real_descriptor import *
from . import hog_descriptor


# TODO: Put somewhere better

def load_descriptor(settings):
    des_name = settings['detector']['descriptor']
    descriptor_filename = settings[des_name].get('file')
    detector_class = gv.Detector.getclass(settings['detector'].get('type', 'binary'))
    descriptor_cls = detector_class.DESCRIPTOR.getclass(des_name)
    if descriptor_filename is None:
        # If there is no descriptor filename, we'll just build it from the settings
        print settings[des_name]
        descriptor = descriptor_cls.load_from_dict(settings[des_name])
    else:
        descriptor = descriptor_cls.load(descriptor_filename)
    return descriptor


def load_binary_descriptor(settings):
    return load_descriptor(gv.BinaryDescriptor, settings)

def load_real_descriptor(settings):
    return load_descriptor(gv.RealDescriptor, settings)

import time

class Timer(object):
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        print "TIMER {0}: {1} s".format(self.name, self.end - self.start)
