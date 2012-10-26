
import os.path
def rel(x):
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), x)

SETTINGS = dict( 
    src_dir='/Users/slimgee/git/data/bike2',
    dst_dir='/Users/slimgee/git/data/afew',
    img_output_dir='/Users/slimgee/git/data/output/images',
    anno_output_dir='/Users/slimgee/git/data/output/annotations',
    xml_template=rel('template.xml'),
    index_file='/Users/slimgee/git/data/output/traingen.txt'
)
