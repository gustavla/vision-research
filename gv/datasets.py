import gv

def load_file(contest, img_id, obj_class=None):
    if contest.startswith('uiuc'):
        return gv.uiuc.load_testing_file(img_id, single_scale==(contest=='uiuc'))
    if contest.startswith('voc'):
        return gv.voc.load_file(obj_class, img_id) 
