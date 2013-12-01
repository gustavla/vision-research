from __future__ import division
import os.path
import numpy as np
import re
import gv
from .datasets import ImgFile

def _get_path():
    try:
        return os.environ["CUSTOM_DIR"]
    except KeyError:
        raise Exception("Please set the environment variable CUSTOM_DIR")

def _img_path(dataset, img_id):
    return os.path.join(_get_path(), dataset, 'test-{0}.png'.format(img_id))

def _convert_line_to_file(dataset, line):
    v = line.split(':')
    img_id = int(v[0])
    bbs = [] 
    rx = re.compile(r'\((-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\)')

    matches = rx.findall(v[1])
    for match in matches:
        bb = tuple(map(int, match))
        bbobj = gv.bb.DetectionBB(box=bb)
        bbs.append(bbobj)

    return ImgFile(path=_img_path(dataset, img_id), boxes=bbs, img_id=img_id)

def _open_box_file(dataset):
    txt_path = os.path.join(_get_path(), '{0}.txt'.format(dataset))
    return open(txt_path)

def load_testing_files(dataset):
    files = []
    for line in _open_box_file(dataset):
        if line.strip() != "":
            fileobj = _convert_line_to_file(dataset, line)
            files.append(fileobj)
    tot = sum([len(f.boxes) for f in files])
    return files, tot

def load_testing_file(dataset, img_id):
    files, tot = load_testing_files(dataset)
    f = filter(lambda x: x.img_id==img_id, files)[0]
    return f
