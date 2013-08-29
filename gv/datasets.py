import gv
from collections import namedtuple
import amitgroup as ag

ImgFile = namedtuple('ImgFile', ['path', 'boxes', 'img_id'])

def contests():
    return ('voc-val', 'voc-test', 'voc-profile', 'voc-profile2', 'voc-profile3', 'voc-profile4', 'voc-profile5', 'voc-easy', 'voc-fronts', 'voc-fronts-negs',
            'uiuc', 'uiuc-multiscale', 
            'custom-cad-profile', 'custom-cad-all', 'custom-cad-all-shuffled')

def datasets():
    return ('none', 'voc', 'uiuc', 'uiuc-multiscale', 
            'custom-cad-profile', 'custom-cad-all', 'custom-cad-all-shuffled')

def load_files(contest, obj_class):
    if contest == 'voc-val':
        files, tot = gv.voc.load_files(obj_class, dataset='val')
    elif contest == 'voc-test':
        files, tot = gv.voc.load_files(obj_class, dataset='test')
    elif contest == 'voc-profile':
        files, tot = gv.voc.load_files(obj_class, dataset='profile')
    elif contest == 'voc-profile2':
        files, tot = gv.voc.load_files(obj_class, dataset='profile2')
    elif contest == 'voc-profile3':
        files, tot = gv.voc.load_files(obj_class, dataset='profile3')
    elif contest == 'voc-profile4':
        files, tot = gv.voc.load_files(obj_class, dataset='profile4')
    elif contest == 'voc-profile5':
        files, tot = gv.voc.load_files(obj_class, dataset='profile5')
    elif contest == 'voc-easy':
        files, tot = gv.voc.load_files(obj_class, dataset='easy')
    elif contest == 'voc-fronts':
        files, tot = gv.voc.load_files(obj_class, dataset='fronts')
    elif contest == 'voc-fronts-negs':
        files, tot = gv.voc.load_files(obj_class, dataset='fronts-negs')
    elif contest == 'uiuc':
        files, tot = gv.uiuc.load_testing_files()
    elif contest == 'uiuc-multiscale':
        files, tot = gv.uiuc.load_testing_files(single_scale=False)
    elif contest.startswith('custom'):
        name = contest[len('custom-'):]
        files, tot = gv.custom.load_testing_files(name)
    else:
        raise ValueError("Contest does not exist: {0}".format(contest))
    return files, tot

def load_file(contest, img_id, obj_class=None, path=None):
    if contest.startswith('uiuc'):
        return gv.uiuc.load_testing_file(img_id, single_scale=(contest=='uiuc'))
    elif contest.startswith('voc'):
        return gv.voc.load_file(obj_class, img_id) 
    elif contest.startswith('custom'):
        name = contest[len('custom-'):]
        return gv.custom.load_testing_file(name, img_id)
    elif contests == 'none':
        assert path is not None 
        return ImgFile(path=path, boxes=[], img_id=-1)

def extract_features_from_bbobj(bbobj, detector, contest, obj_class, kernel_shape):
    """
    Retrieves features from a DetectionBB object 
    """
    #bb = (det['top'], det['left'], det['bottom'], det['right'])
    #k = det['mixcomp']
    #m = det['bkgcomp']
    #bbobj = gv.bb.DetectionBB(bb, score=det['confidence'], confidence=det['confidence'], mixcomp=k, correct=det['correct'])

    img_id = bbobj.img_id
    #img_id = det['img_id']
    fileobj = load_file(contest, img_id, obj_class=obj_class)

    im = gv.img.load_image(fileobj.path) 
    im = gv.img.asgray(im)
    im = gv.img.resize_with_factor_new(im, 1/bbobj.scale)

    #kern = kernels[k][m]
    #bkg = all_bkg[k][m]
    #kern = np.clip(kern, eps, 1 - eps)
    #bkg = np.clip(bkg, eps, 1 - eps)

    d0, d1 = kernel_shape #kern.shape[:2] 

    psize = detector.settings['subsample_size']
    radii = detector.settings['spread_radii']
    
    feats = detector.descriptor.extract_features(im, dict(spread_radii=radii, subsample_size=psize, preserve_size=False))

    i0, j0 = bbobj.index_pos
    pad = max(-min(0, i0), -min(0, j0), max(0, i0+d0 - feats.shape[0]), max(0, j0+d1 - feats.shape[1]))

    feats = ag.util.zeropad(feats, (pad, pad, 0))
    X = feats[pad+i0:pad+i0+d0, pad+j0:pad+j0+d1]
    return X 
