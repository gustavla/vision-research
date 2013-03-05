from __future__ import division
import os.path
import numpy as np
import skimage.data
import re
import gv
from .voc import ImgFile

_SIZE = (40, 100)

def _get_path():
    try:
        return os.environ["UIUC_DIR"]
    except KeyError:
        raise Exception("Please set the environment variable UIUC_DIR")
    
def _img_path(img_id):
    return os.path.join(_get_path(), 'TestImages', 'test-{0}.pgm'.format(img_id))

def _convert_line_to_file(line):
    v = line.split(':')
    img_id = int(v[0])
    box_strs = v[1].strip().split(' ')
    bbs = [] 
    size = _SIZE
    rx = re.compile(r'\((-?\d+),(-?\d+)\)')
    for s in box_strs:
        match = rx.match(s)
        pos = (int(match.group(1)), int(match.group(2)))
        bb = (pos[0], pos[1], pos[0]+size[0], pos[1]+size[1])
        bbobj = gv.bb.DetectionBB(box=bb)
        bbs.append(bbobj)

    return ImgFile(path=_img_path(img_id), boxes=bbs, img_id=img_id)

def _open_box_file():
    txt_path = os.path.join(_get_path(), 'trueLocations.txt')
    return open(txt_path)

def load_testing_files(single_scale=True):
    files = []
    for line in _open_box_file():
        fileobj = _convert_line_to_file(line)
        files.append(fileobj)
    tot = sum([len(f.boxes) for f in files])
    return files, tot

def load_testing_file(img_id, single_scale=True):
    files, tot = load_testing_files()
    f = filter(lambda x: x.img_id==img_id, files)[0]
    return f
