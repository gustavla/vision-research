from __future__ import division
import os.path
import numpy as np
from xml.dom.minidom import parse
from collections import namedtuple
import gv

ImgFile = namedtuple('ImgFile', ['path', 'boxes', 'img_id'])

def _get_text(nodelist):
    rc = []
    for node in nodelist:
        if node.nodeType == node.TEXT_NODE:
            rc.append(node.data)
    return ''.join(rc)

def image_from_bounding_box(im, bb):
    if isinstance(bb, gv.bb.DetectionBB):
        bb = bb.box

    #if gv.bb.box_sticks_out(bb, (0, 0)+im.shape[:2]):
        # 

    #else:
    return im[bb[0]:bb[2], bb[1]:bb[3]]


def load_file(class_name, img_id, load_boxes=True):
    img_path = os.path.join(os.environ['VOC_DIR'], 'JPEGImages', '{0:06}.jpg'.format(img_id))
    bbs = []
    if load_boxes: 
        # Load bounding boxes of object
        xml_path = os.path.join(os.environ['VOC_DIR'], 'Annotations', '{0:06}.xml'.format(img_id))
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
                bb = tuple([int(_get_text(bndbox_obj.getElementsByTagName(s)[0].childNodes)) - 1 \
                        for s in ('ymin', 'xmin', 'ymax', 'xmax')])
                bbobj = gv.bb.DetectionBB(box=bb, difficult=difficult, truncated=truncated)
                bbs.append(bbobj)

    fileobj = ImgFile(path=img_path, boxes=bbs, img_id=img_id)
    return fileobj

def load_specific_files(class_name, img_ids, has_objects=None, padding=0):
    """img_ids and has_objects should be lists of equal length"""
    N = len(img_ids) 

    files = [] 
    hasobject = 1 
    for i in xrange(N):
        img_id = img_ids[i]
        if has_objects is not None:
            hasobject = has_objects[i] 
        fileobj = load_file(class_name, img_id, load_boxes=(hasobject == 1))
        files.append(fileobj)

    # Get the total count
    tot = sum([len(f.boxes) for f in files])

    return files, tot

def load_files(class_name, dataset='train'):
    path = os.path.join(os.environ['VOC_DIR'], 'ImageSets', 'Main', '{0}_{1}.txt'.format(class_name, dataset))

    f = np.genfromtxt(path, dtype=int) 
    N = f.shape[0]
    img_ids = f[:,0]
    has_objects = f[:,1] 
    return load_specific_files(class_name, img_ids, has_objects)

def _load_images(objfiles, tot, size, padding=0):
    bbs = []
    originals = []
    for objfile in objfiles:
        for bbobj in objfile.boxes:
            # Only non-truncated and non-difficult ones
            if not bbobj.truncated and not bbobj.difficult:
                im = gv.img.load_image(objfile.path)
                bbsquare = bbobj.box #gv.bb.expand_to_square(bbobj.box)
                #bbsquare = gv.bb.expand_to_square(bbobj.box)
                # Resize padding with a factor, so that the end image will have that
                # padding.
                factor = max(size) / max(bbsquare[4]-bbsquare[1], bbsquare[2]-bbsquare[0])
        
                # Resize the big image
                img_resized = gv.img.resize_with_factor(im, factor) 


                bbsquare_padded = gv.bb.inflate(bbsquare, int(round(padding * factor)))
                bbim = (0, 0)+im.shape[:2]
                if gv.bb.box_sticks_out(bbsquare_padded, bbim):
                    continue
    
                bbsquare_resized = tuple([bbsquare_padded[i] * factor for i in xrange(4)])

                padded_size = (size[0] + 2*padding, size[1] + 2*padding)
                if 1:
                    image_resized = image_from_bounding_box(img_resized, bbsquare_resized)
                else:
                    im_patch = image_from_bounding_box(im, bbsquare_padded)
                    image_resized = gv.img.resize(im_patch, padded_size)

                #images.append(image_resized)  
                originals.append(img_resized)
                bbs.append(bbsquare_resized)

    return originals, bbs

def load_object_images_of_size_from_list(class_name, size, img_ids, padding=0):
    objfiles, tot = load_specific_files(class_name, img_ids)
    return _load_images(objfiles, tot, size, padding=padding)

def load_negative_images_of_size(class_name, size, dataset='train', count=10, padding=0):
    padded_size = (size[0]+2*padding, size[1]+2*padding)
    bbs = []
    originals = []
    i = 1
    while True: 
        objfile = load_file(class_name, i, load_boxes=False)
        im = gv.img.load_image(objfile.path)

        # Randomize factor
        factor = np.random.uniform(0.3, 1.0)

        im_resized = gv.img.resize_with_factor(im, factor)
    
        shift = [im_resized.shape[i] - padded_size[i] for i in xrange(2)]
        if min(shift) > 0: 
            i, j = [np.random.randint(shift[i]) for i in xrange(2)]

            # Extract image
            im_patch = im_resized[i:i+padded_size[0], j:j+padded_size[1]]
            #images.append(im_patch)
            bb = (i, j, i+padded_size[0], j+padded_size[1])
            bbs.append(bb)
            originals.append(im_resized)
        
        i += 1
        if len(bbs) >= count:
            break

    return originals, bbs

def load_object_images_of_size(class_name, size, dataset='train'):
    objfiles, tot = load_files(class_name, dataset=dataset)
    return _load_images(objfiles, tot, size)

                 

if __name__ == '__main__':
    pass
    #files, tot = load_files('bicycle')
    #print tot
    #print files[:20]
