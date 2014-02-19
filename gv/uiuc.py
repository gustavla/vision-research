from __future__ import division
import os.path
import numpy as np
import re
import gv
from .datasets import ImgFile

_SIZE = (40, 100)

def _get_path(env='UIUC_DIR'):
    try:
        return os.environ[env]
    except KeyError:
        raise Exception("Please set the environment variable {}".format(env))
    
def _img_path(img_id, single_scale=True, env='UIUC_DIR'):
    if single_scale:
        return os.path.join(_get_path(env), 'TestImages', 'test-{0}.pgm'.format(img_id))
    else:
        return os.path.join(_get_path(env), 'TestImages_Scale', 'test-{0}.pgm'.format(img_id))

def _convert_line_to_file(line, single_scale=True, env='UIUC_DIR'):
    v = line.split(':')
    img_id = int(v[0])
    box_strs = v[1].strip().split(' ')
    bbs = [] 
    size = _SIZE
    if single_scale:
        rx = re.compile(r'\((-?\d+),(-?\d+)\)')
    else:
        rx = re.compile(r'\((-?\d+),(-?\d+),(-?\d+)\)')

    for s in box_strs:
        match = rx.match(s)
        pos = (int(match.group(1)), int(match.group(2)))
        if not single_scale:
            w = int(match.group(3))
            size = (int(_SIZE[0]/_SIZE[1] * w), w)
        bb = (pos[0], pos[1], pos[0]+size[0], pos[1]+size[1])
        # Note! We are shrinking the bounding boxes a bit to make them tighter.
        # The original ones contain non-car space around them, so this is making
        # them closer to the Pascal VOC way, where they are tightly around the car.
        bb = gv.bb.inflate(bb, -4)

        bbobj = gv.bb.DetectionBB(box=bb)
        bbs.append(bbobj)

    return ImgFile(path=_img_path(img_id, single_scale, env=env), boxes=bbs, img_id=img_id)

def _open_box_file(single_scale=True, env='UIUC_DIR'):
    if single_scale:
        txt_path = os.path.join(_get_path(env), 'trueLocations.txt')
    else:
        txt_path = os.path.join(_get_path(env), 'trueLocations_Scale.txt')
    return open(txt_path)

def load_testing_files(single_scale=True, env='UIUC_DIR'):
    files = []
    for line in _open_box_file(single_scale, env=env):
        if line.strip() != "":
            fileobj = _convert_line_to_file(line, single_scale=single_scale, env=env)
            files.append(fileobj)
    tot = sum([len(f.boxes) for f in files])
    return files, tot

def load_testing_file(img_id, single_scale=True, env='UIUC_DIR'):
    img_id = int(img_id)
    files, tot = load_testing_files(single_scale, env=env)
    f = filter(lambda x: x.img_id==img_id, files)[0]
    return f
