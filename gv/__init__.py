from __future__ import absolute_import
from .detector import *
from . import img
from . import voc
from . import uiuc
from . import bb
from . import sub 
from . import rescalc
from . import datasets
from .ndfeature import ndfeature

from .binary_descriptor import *
from . import edge_descriptor
from . import parts_descriptor


# TODO: Put somewhere better

def load_descriptor(settings):
    des_name = settings['detector']['descriptor']
    descriptor_filename = settings[des_name].get('file')
    descriptor_cls = gv.BinaryDescriptor.getclass(des_name)
    if descriptor_filename is None:
        # If there is no descriptor filename, we'll just build it from the settings
        print settings[des_name]
        descriptor = descriptor_cls.load_from_dict(settings[des_name])
    else:
        descriptor = descriptor_cls.load(descriptor_filename)
    return descriptor

import time

class Timer(object):
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        print "TIMER {0}: {1} s".format(self.name, self.end - self.start)
