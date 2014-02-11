from __future__ import division
from collections import namedtuple
import numpy as np

def _repr(tup):
    return "("+(", ".join(["{"+str(i)+":0.1f}" for i in xrange(4)])).format(*tup)+")"

class DetectionBB(object):
    def __init__(self, box, score=0.0, confidence=0.5, correct=False, difficult=False, index_pos=(0, 0), truncated=False, scale=0, mixcomp=None, plusscore=0.0, score0=0.0, score1=0.0, bkgcomp=None, img_id=None, image=None, X=None, overlap=None):
        self.score = score
        self.score0 = score0
        self.score1 = score1
        self.box = box
        self.confidence = confidence
        self.correct = correct
        self.difficult = difficult
        self.truncated = truncated
        self.scale = scale
        self.mixcomp = mixcomp
        self.bkgcomp = bkgcomp
        self.plusscore = plusscore
        self.index_pos = index_pos
        self.img_id = img_id
        self.image = image
        self.X = X
        self.overlap = overlap

    def __repr__(self):
        try:
            return "DetectionBB(score={score:.2f}, box={box}, correct={correct}, confidence={confidence:.2f}, mixcomp={mixcomp}, bkgcomp={bkgcomp}, scale={scale})".format(
                score=self.score, 
                box=_repr(self.box), 
                correct=self.correct, 
                confidence=self.confidence, 
                mixcomp=self.mixcomp, 
                bkgcomp=self.bkgcomp,
                scale=self.scale)
        except:
            import pdb; pdb.set_trace()

    def __cmp__(self, b):
        return cmp(self.score, b.score)

def intersection(bb1, bb2):
    return (max(bb1[0], bb2[0]), max(bb1[1], bb2[1]),
            min(bb1[2], bb2[2]), min(bb1[3], bb2[3]))

def area(bb):
    return max(0, (bb[2] - bb[0])) * max(0, (bb[3] - bb[1]))
    #return (bb[2] - bb[0]) * (bb[3] - bb[1])
    
def size(bb):
    return (bb[2] - bb[0], bb[3] - bb[1])

def union_area(bb1, bb2):
    return area(bb1) + area(bb2) - area(intersection(bb1, bb2))

def fraction_metric(bb1, bb2):
    return area(intersection(bb1, bb2)) / union_area(bb1, bb2)

def box_sticks_out(bb_smaller, bb_bigger):
    return area(bb_bigger) != union_area(bb_bigger, bb_smaller)

def inflate(bb, amount):
    return (bb[0] - amount, bb[1] - amount, bb[2] + amount, bb[3] + amount)

def inflate2(bb, amounts):
    return (bb[0] - amounts[0], bb[1] - amounts[1], bb[2] + amounts[0], bb[3] + amounts[1])

def center(bb):
    return ((bb[0]+bb[2])//2, (bb[1]+bb[3])//2)

def create(center=None, size=None):
    assert size is not None
    if center is None:
        center = (0, 0)
    half_size = [s//2 for s in size]
    return (center[0]-half_size[0], center[1]-half_size[1],
            center[1]-half_size[0]+size[0], center[1]-half_size[1]+size[1])

def expand_to_square(bb):
    """Expands a bounding box to square"""
    diffs = [bb[2]-bb[0], bb[3]-bb[1]]
    smaller_axis = np.argmin(diffs)
    mn, mx = min(diffs), max(diffs)
    w = (mx - mn)//2
    if smaller_axis == 0:
        bb = (bb[0]-w, bb[1], bb[2]-w+(mx-mn), bb[3])
    else:
        bb = (bb[0], bb[1]-w, bb[2], bb[3]-w+(mx-mn))
    return bb 
