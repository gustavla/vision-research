import os.path
import numpy as np
from xml.dom.minidom import parse
from collections import namedtuple
import gv

ImgFile = namedtuple('ImgFile', ['path', 'boxes'])

def _get_text(nodelist):
    rc = []
    for node in nodelist:
        if node.nodeType == node.TEXT_NODE:
            rc.append(node.data)
    return ''.join(rc)

def load_training_file(VOCSETTINGS, class_name, img_id, load_boxes=True, dataset='test'):
    img_path = os.path.join(VOCSETTINGS['path'], 'JPEGImages', '{0:06}.jpg'.format(img_id))
    bbs = []
    if load_boxes: 
        # Load bounding boxes of object
        xml_path = os.path.join(VOCSETTINGS['path'], 'Annotations', '{0:06}.xml'.format(img_id))
        dom = parse(xml_path)
        # Get all objects
        objs = dom.getElementsByTagName('object')
        for obj in objs:
            # Check what kind of object
            name = _get_text(obj.getElementsByTagName('name')[0].childNodes)
            if name == class_name:
                truncated = bool(int(_get_text(obj.getElementsByTagName('truncated')[0].childNodes)))
                difficult = bool(int(_get_text(obj.getElementsByTagName('difficult')[0].childNodes)))
                bndbox_obj = obj.getElementsByTagName('bndbox')[0] 
                # Note: -1 is taken because they use 1-base indexing
                bb = tuple([int(_get_text(bndbox_obj.getElementsByTagName(s)[0].childNodes)) - 1 
                        for s in 'ymin', 'xmin', 'ymax', 'xmax'])
                bbobj = gv.bb.DetectionBB(box=bb, difficult=difficult)
                bbs.append(bbobj)

    fileobj = ImgFile(path=img_path, boxes=bbs)
    return fileobj

def load_training_files(VOCSETTINGS, class_name, dataset='test'):
    path = os.path.join(VOCSETTINGS['path'], 'ImageSets', 'Main', '{0}_{1}.txt'.format(class_name, dataset))

    f = np.genfromtxt(path, dtype='i') 
    N = f.shape[0]

    files = [] 
    for i in xrange(N):
        img_id, hasobject = f[i]
        fileobj = load_training_file(VOCSETTINGS, class_name, img_id, load_boxes=(hasobject == 1), dataset=dataset)
        files.append(fileobj)

    # Get the total count
    tot = sum([len(f.boxes) for f in files])

    return files, tot

if __name__ == '__main__':
    pass
    #files, tot = load_training_files('bicycle')
    #print tot
    #print files[:20]
