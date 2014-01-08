from __future__ import division
import os.path
import numpy as np
import re
import gv
from .datasets import ImgFile

_SIZE = (40, 100)

def _get_path():
    try:
        return os.environ["UIUC_DIR"]
    except KeyError:
        raise Exception("Please set the environment variable UIUC_DIR")
    
def _img_path(img_id, single_scale=True):
    if single_scale:
        return os.path.join(_get_path(), 'TestImages', 'test-{0}.pgm'.format(img_id))
    else:
        return os.path.join(_get_path(), 'TestImages_Scale', 'test-{0}.pgm'.format(img_id))

def _convert_line_to_file(line, single_scale=True):
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
        bbobj = gv.bb.DetectionBB(box=bb)
        bbs.append(bbobj)

    return ImgFile(path=_img_path(img_id, single_scale), boxes=bbs, img_id=img_id)

def _open_box_file(single_scale=True):
    if single_scale:
        txt_path = os.path.join(_get_path(), 'trueLocations.txt')
    else:
        txt_path = os.path.join(_get_path(), 'trueLocations_Scale.txt')
    return open(txt_path)

def load_testing_files(single_scale=True):
    files = []
    for line in _open_box_file(single_scale):
        if line.strip() != "":
            fileobj = _convert_line_to_file(line, single_scale=single_scale)
            files.append(fileobj)
    tot = sum([len(f.boxes) for f in files])
    return files, tot

def load_testing_file(img_id, single_scale=True):
    img_id = int(img_id)
    files, tot = load_testing_files(single_scale)
    f = filter(lambda x: x.img_id==img_id, files)[0]
    return f
