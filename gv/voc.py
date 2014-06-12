
import os.path
import numpy as np
from xml.dom.minidom import parse
import gv
from .datasets import ImgFile

XML_TEMPLATE = """<annotation>
	<folder>VOC2007</folder>
	<filename>{filename}</filename>
	<source>
		<database>CAD Project Generation</database>
		<annotation>CAD Project</annotation>
		<image>generated</image>
		<flickrid>?</flickrid>
	</source>
	<owner>
		<flickrid>?</flickrid>
		<name>?</name>
	</owner>
	<size>
		<width>{width}</width>
		<height>{height}</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>car</name>
		<pose></pose>
		<truncated>{truncated}</truncated>
		<difficult>{difficult}</difficult>
		<bndbox>
			<xmin>{xmin}</xmin>
			<ymin>{ymin}</ymin>
			<xmax>{xmax}</xmax>
			<ymax>{ymax}</ymax>
		</bndbox>
	</object>
</annotation>
"""

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

def gen_negative_files(excluding_class, contest='train'):
    path = os.path.join(os.environ['VOC_DIR'], 'ImageSets', 'Main', '{0}_{1}.txt'.format(excluding_class, contest))
    for line in open(path):
        img_id, s = line.split() 
        if int(s) == -1:
            yield load_file(excluding_class, int(img_id), load_boxes=False)

def load_xml_file(xml_path, class_name=None, poses=None):
    dom = parse(xml_path)

    # Get image size info
    sizetag = dom.getElementsByTagName('size')[0]

    width = int(_get_text(sizetag.getElementsByTagName('width')[0].childNodes))
    height = int(_get_text(sizetag.getElementsByTagName('height')[0].childNodes))

    # Get all objects
    objs = dom.getElementsByTagName('object')
    bbs = []
    for obj in objs:
        # Check what kind of object
        name = _get_text(obj.getElementsByTagName('name')[0].childNodes)
        pose = _get_text(obj.getElementsByTagName('pose')[0].childNodes)
        pose_ok = poses is None or pose in poses
        if (name == class_name or class_name is None) and pose_ok:
            truncated = bool(int(_get_text(obj.getElementsByTagName('truncated')[0].childNodes)))
            difficult = bool(int(_get_text(obj.getElementsByTagName('difficult')[0].childNodes)))
            bndbox_obj = obj.getElementsByTagName('bndbox')[0] 
            # Note: -1 is taken because they use 1-base indexing
            bb = tuple([float(_get_text(bndbox_obj.getElementsByTagName(s)[0].childNodes)) - 1 \
                    for s in ('ymin', 'xmin', 'ymax', 'xmax')])
            bbobj = gv.bb.DetectionBB(box=bb, difficult=difficult, truncated=truncated)
            bbs.append(bbobj)
    return bbs, (height, width)


def load_file(class_name, img_id, load_boxes=True, poses=None, image_dir=None, anno_dir=None):
    if image_dir is None:
        image_dir = os.path.join(os.environ['VOC_DIR'], 'JPEGImages')
    if anno_dir is None:
        anno_dir = os.path.join(os.environ['VOC_DIR'], 'Annotations')

    img_id = int(img_id)
    for ext in 'jpg', 'png': 
        img_path = os.path.join(image_dir, '{0:06}.{1}'.format(img_id, ext))
        if os.path.isfile(img_path):
            break
    bbs = []
    if load_boxes: 
        # Load bounding boxes of object
        xml_path = os.path.join(anno_dir, '{0:06}.xml'.format(img_id))
        bbs, img_size = load_xml_file(xml_path, class_name=class_name, poses=poses)
    else:
        img_size = None

    fileobj = ImgFile(path=img_path, boxes=bbs, img_id=img_id, img_size=img_size)
    return fileobj

def load_specific_files(class_name, img_ids, has_objects=None, padding=0, poses=None, image_dir=None, anno_dir=None):
    """img_ids and has_objects should be lists of equal length"""
    N = len(img_ids) 

    files = [] 
    hasobject = 1 
    for i in range(N):
        img_id = img_ids[i]
        if has_objects is not None:
            hasobject = has_objects[i] 
        fileobj = load_file(class_name, img_id, load_boxes=(hasobject == 1), poses=poses, image_dir=image_dir, anno_dir=anno_dir)
        files.append(fileobj)

    # Get the total count
    tot = sum([len(f.boxes) for f in files])

    return files, tot

_TEST_NEGS = [
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

_NEGS = [1, 2, 5, 6, 8, 9, 10, 11, 13, 15, 16, 30, 31, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 70, 73, 75, 76, 100]
_NEGS2 = list(range(104, 130))
_NEGS3 = list(range(138, 141+1)) + list(range(143, 151+1)) + list(range(162, 168+1)) + [170, 171] + list(range(173, 179+1)) + list(range(181, 187+1)) + list(range(191, 196+1)) + list(range(198, 209+1)) + list(range(211, 219+1))
_VOC_OTHER_PROFILES = [3971, 3973, 4830, 4962, 5199, 5350, 5749, 6062, 7843, 8057, 8329, 8429, 8461, 9029, 9671, 9913]
_VOC_PROFILES = [153, 220, 263, 317, 522, 871, 1060, 1119, 1662, 2182, 3790, 3936]
_VOC_PROFILES2 = _VOC_PROFILES + _NEGS
_VOC_PROFILES3 = _VOC_PROFILES + [334, 2436, 4231, 5020, 7279, 8044, 7819, 8483, 8891, 8929, 9078, 9409, 9959] + _NEGS + _NEGS2 + _NEGS3
_VOC_PROFILES4 = _VOC_OTHER_PROFILES + _NEGS + _NEGS2 + _NEGS3
_VOC_PROFILES5 = _VOC_PROFILES + _TEST_NEGS  # Does not have training data negatives in it!
_VOC_EASY_NONPROFILES = [26, 1237, 1334, 1488, 1494, 1576, 2153, 2178, 2247, 2534]
_VOC_FRONTBACKS = [74, 152, 240, 252, 271, 313, 341, 361, 390, 471, 505, 580, 586, 593, 602, 607, 646, 649, 1003, 1111, 1252]
_VOC_FRONTBACKS_NEGS = _VOC_FRONTBACKS + _TEST_NEGS

_VOC_SIDES = [
    4, 71, 82, 103, 135, 137, 152, 172, 293, 301, 358, 415, 585, 679, 693, 715, 721, 
    724, 736, 801, 881, 1005, 1034, 1280, 1283, 1356, 1369, 1379, 1382, 1394, 1422, 1491,
    1511, 1525, 1535, 1550, 1552, 1560, 1572, 1700, 1770, 1838, 1923, 1924, 1935, 1951, 1991,
    2154, 2232, 2242, 2271, 2331, 2346, 2416, 2484, 2531, 2703, 2733, 2840, 2900, 2927, 3006,
    3033, 3046, 3055, 3070, 3109, 3143, 3276, 3306, 3348, 3357, 3364, 3375
]

def load_files(class_name, dataset='train', poses=None, image_dir=None, anno_dir=None):
    if dataset == 'profile':
        return load_specific_files(class_name, _VOC_PROFILES, poses=poses, image_dir=image_dir, anno_dir=anno_dir)
    elif dataset == 'profile2':
        return load_specific_files(class_name, _VOC_PROFILES2, poses=poses, image_dir=image_dir, anno_dir=anno_dir)
    elif dataset == 'profile3':
        return load_specific_files(class_name, _VOC_PROFILES3, poses=poses, image_dir=image_dir, anno_dir=anno_dir)
    elif dataset == 'profile4':
        return load_specific_files(class_name, _VOC_PROFILES4, poses=poses, image_dir=image_dir, anno_dir=anno_dir)
    elif dataset == 'profile5':
        return load_specific_files(class_name, _VOC_PROFILES5, poses=poses, image_dir=image_dir, anno_dir=anno_dir)
    elif dataset == 'easy':
        return load_specific_files(class_name, _VOC_PROFILES + _VOC_EASY_NONPROFILES, poses=poses, image_dir=image_dir, anno_dir=anno_dir)
    elif dataset == 'fronts':
        return load_specific_files(class_name, _VOC_FRONTBACKS, poses=poses, image_dir=image_dir, anno_dir=anno_dir)
    elif dataset == 'fronts-negs':
        return load_specific_files(class_name, _VOC_FRONTBACKS_NEGS, poses=poses, image_dir=image_dir, anno_dir=anno_dir)
    elif dataset == 'sides':
        return load_specific_files(class_name, _VOC_SIDES, poses=poses, image_dir=image_dir, anno_dir=anno_dir)

    path = os.path.join(os.environ['VOC_DIR'], 'ImageSets', 'Main', '{0}_{1}.txt'.format(class_name, dataset))

    f = np.genfromtxt(path, dtype=int) 
    N = f.shape[0]
    img_ids = f[:,0]
    has_objects = f[:,1] 
    return load_specific_files(class_name, img_ids, has_objects, poses=poses, image_dir=image_dir, anno_dir=anno_dir)

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
    
                bbsquare_resized = tuple([bbsquare_padded[i] * factor for i in range(4)])

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
    
        shift = [im_resized.shape[i] - padded_size[i] for i in range(2)]
        if min(shift) > 0: 
            i, j = [np.random.randint(shift[i]) for i in range(2)]

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

def save_file(path, fileobj):
    fn = '{}.png'.format(fileobj.img_id)
    assert len(fileobj.boxes) == 1, 'Can only save files with one object'
    bbobj = fileobj.boxes[0]
    bb = bbobj.box

    content = XML_TEMPLATE.format(
        filename=fn,
        width=fileobj.img_size[1],
        height=fileobj.img_size[0],
        xmin=int(bb[1]+1),
        ymin=int(bb[0]+1),
        xmax=int(bb[3]+1),
        ymax=int(bb[2]+1),
        truncated=int(bbobj.truncated),
        difficult=int(bbobj.difficult),
    )

    open(os.path.join(path, '{}.xml'.format(fileobj.img_id)), 'w').write(content)


def save_files(path, fileobjs):
    for fileobj in fileobjs:
        save_file(path, fileobj)

if __name__ == '__main__':
    pass
    #files, tot = load_files('bicycle')
    #print tot
    #print files[:20]
