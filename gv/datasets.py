import gv
from collections import namedtuple

ImgFile = namedtuple('ImgFile', ['path', 'boxes', 'img_id'])

def contests():
    return ('voc-val', 'voc-profile', 'voc-profile2', 'voc-profile3', 'voc-profile4', 'voc-easy', 
            'uiuc', 'uiuc-multiscale', 
            'custom-cad-profile')

def datasets():
    return ('none', 'voc', 'uiuc', 'uiuc-multiscale', 
            'custom-cad-profile')

def load_files(contest, obj_class):
    if contest == 'voc-val':
        files, tot = gv.voc.load_files(obj_class, dataset='val')
    elif contest == 'voc-profile':
        files, tot = gv.voc.load_files(obj_class, dataset='profile')
    elif contest == 'voc-profile2':
        files, tot = gv.voc.load_files(obj_class, dataset='profile2')
    elif contest == 'voc-profile3':
        files, tot = gv.voc.load_files(obj_class, dataset='profile3')
    elif contest == 'voc-profile4':
        files, tot = gv.voc.load_files(obj_class, dataset='profile4')
    elif contest == 'voc-easy':
        files, tot = gv.voc.load_files(obj_class, dataset='easy')
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
