
import os.path
def rel(x):
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), x)

if 0:
    SETTINGS = dict( 
        patches_dir='/var/tmp/matlab/VOC2007/VOCdevkit/VOC2007/JPEGImages',
        src_dir='/var/tmp/matlab/newbike',
        dst_dir='/var/tmp/matlab/VOC2007/VOCdevkit/VOC2007/JPEGImages',
        img_output_dir='/var/tmp/matlab/VOC2007/VOCdevkit/VOC2007/JPEGImages',
        anno_output_dir='/var/tmp/matlab/VOC2007/VOCdevkit/VOC2007/Annotations',
        xml_template=rel('template.xml'),
        index_file='/var/tmp/matlab/VOC2007/VOCdevkit/VOC2007/ImageSets/Main/traingen.txt'
    )
else: 
    SETTINGS = dict( 
        src_dir=os.path.expandvars('$DATA_DIR/ncad01'),
        #src_tight_dir='/Users/slimgee/git/data/newbiketight',
        #dst_dir=os.path.expandvars('$DATA_DIR/generated/images'),# '/Users/slimgee/git/data/afew',
        dst_dir=os.path.expandvars('$VOC_DIR/JPEGImages'),
        #posed_dir='/Users/slimgee/git/data/posed_bikes',
        img_output_dir=os.path.expandvars('$DATA_DIR/generated/images'),
        anno_output_dir=os.path.expandvars('$DATA_DIR/generated/anno'),
        xml_template=rel('template.xml'),
        #index_file='/Users/slimgee/git/data/output/traingen.txt'
        index_file=os.path.expandvars('$DATA_DIR/generated/traingen.txt'),
    )
