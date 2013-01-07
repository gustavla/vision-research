from __future__ import division
from collections import namedtuple

class DetectionBB(object):
    def __init__(self, box, score=0.0, confidence=0.5, correct=False, difficult=False):
        self.score = score
        self.box = box
        self.confidence = confidence
        self.correct = correct
        self.difficult = difficult

    def __repr__(self):
        return "DetectionBB(score={0}, box={1}, correct={2})".format(self.score, self.box, self.correct)

    def __cmp__(self, b):
        return cmp(self.score, b.score)

def intersection(bb1, bb2):
    return (max(bb1[0], bb2[0]), max(bb1[1], bb2[1]),
            min(bb1[2], bb2[2]), min(bb1[3], bb2[3]))

def area(bb):
    return max(0, (bb[2] - bb[0])) * max(0, (bb[3] - bb[1]))

def union_area(bb1, bb2):
    return area(bb1) + area(bb2) - area(intersection(bb1, bb2))

def fraction_metric(bb1, bb2):
    return area(intersection(bb1, bb2)) / union_area(bb1, bb2)

