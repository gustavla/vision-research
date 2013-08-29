from __future__ import division
import os.path
import numpy as np
from xml.dom.minidom import parse
import gv
from .datasets import ImgFile

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

def gen_negative_files(excluding_class):
    path = os.path.join(os.environ['VOC_DIR'], 'ImageSets', 'Main', '{0}_train.txt'.format(excluding_class))
    for line in open(path):
        img_id, s = line.split() 
        if int(s) == -1:
            yield load_file(excluding_class, int(img_id), load_boxes=False)

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

_NEGS = [1, 2, 5, 6, 8, 9, 10, 11, 13, 15, 16, 30, 31, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 70, 73, 75, 76, 100]
_NEGS2 = range(104, 130)
_NEGS3 = range(138, 141+1) + range(143, 151+1) + range(162, 168+1) + [170, 171] + range(173, 179+1) + range(181, 187+1) + range(191, 196+1) + range(198, 209+1) + range(211, 219+1)
_VOC_OTHER_PROFILES = [3971, 3973, 4830, 4962, 5199, 5350, 5749, 6062, 7843, 8057, 8329, 8429, 8461, 9029, 9671, 9913]
_VOC_PROFILES = [153, 220, 263, 317, 522, 871, 1060, 1119, 1662, 2182, 3790, 3936]
_VOC_PROFILES2 = _VOC_PROFILES + _NEGS
_VOC_PROFILES3 = _VOC_PROFILES + [334, 2436, 4231, 5020, 7279, 8044, 7819, 8483, 8891, 8929, 9078, 9409, 9959] + _NEGS + _NEGS2 + _NEGS3
_VOC_PROFILES4 = _VOC_OTHER_PROFILES + _NEGS + _NEGS2 + _NEGS3
_VOC_EASY_NONPROFILES = [26, 1237, 1334, 1488, 1494, 1576, 2153, 2178, 2247, 2534]
_VOC_FRONTBACKS = [74, 152, 240, 252, 271, 313, 341, 361, 390, 471, 505, 580, 586, 593, 602, 607, 646, 649, 1003, 1111, 1252]
_VOC_FRONTBACKS_NEGS = _VOC_FRONTBACKS + [
    1, 2, 3, 6, 8, 10, 11, 13, 15, 18, 
    22, 25, 27, 28, 29, 31, 37, 38, 40, 
    43, 45, 49, 53, 54, 55, 56, 57, 58, 
    59, 62, 67, 67, 68, 69, 70, 75, 76, 
    79, 80, 84, 85, 86, 87, 88, 90, 92, 
    94, 96, 97, 98, 100, 105, 106, 108, 
    111, 114, 115, 116, 119, 124, 126, 127,
    128, 136, 139, 144, 145, 148, 149, 151, 
    155, 157, 160, 166, 167, 168, 175, 176,
    178, 179, 181, 182, 183, 185, 186, 191, 
    195, 196, 199, 201, 202, 204, 205, 206,
    212, 213, 216,
]

def load_files(class_name, dataset='train'):
    if dataset == 'profile':
        return load_specific_files(class_name, _VOC_PROFILES)
    elif dataset == 'profile2':
        return load_specific_files(class_name, _VOC_PROFILES2)
    elif dataset == 'profile3':
        return load_specific_files(class_name, _VOC_PROFILES3)
    elif dataset == 'profile4':
        return load_specific_files(class_name, _VOC_PROFILES4)
    elif dataset == 'easy':
        return load_specific_files(class_name, _VOC_PROFILES + _VOC_EASY_NONPROFILES)
    elif dataset == 'fronts':
        return load_specific_files(class_name, _VOC_FRONTBACKS)
    elif dataset == 'fronts-negs':
        return load_specific_files(class_name, _VOC_FRONTBACKS_NEGS)


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
