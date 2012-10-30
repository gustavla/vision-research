
import os.path
def rel(x):
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), x)

if 0:
    SETTINGS = dict( 
        src_dir='/var/tmp/matlab/bike2',
        dst_dir='/var/tmp/matlab/VOC2007/VOCdevkit/VOC2007/JPEGImages',
        img_output_dir='/var/tmp/matlab/VOC2007/VOCdevkit/VOC2007/JPEGImages',
        anno_output_dir='/var/tmp/matlab/VOC2007/VOCdevkit/VOC2007/Annotations',
        xml_template=rel('template.xml'),
        index_file='/var/tmp/matlab/VOC2007/VOCdevkit/VOC2007/ImageSets/Main/traingen.txt'
    )
else: 
    SETTINGS = dict( 
        src_dir='/Users/slimgee/git/data/newbike',
        dst_dir='/Users/slimgee/git/data/afew',
        posed_dir='/Users/slimgee/git/data/posed_bikes',
        img_output_dir='/Users/slimgee/git/data/output/images',
        anno_output_dir='/Users/slimgee/git/data/output/annotations',
        xml_template=rel('template.xml'),
        index_file='/Users/slimgee/git/data/output/traingen.txt'
    )
