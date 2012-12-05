
if 0:
    SETTINGS = dict( 
        patches_dir='/var/tmp/matlab/VOC2007/VOCdevkit/VOC2007/JPEGImages',
        src_dir='/var/tmp/matlab/bike2',
        dst_dir='/var/tmp/matlab/VOC2007/VOCdevkit/VOC2007/JPEGImages',
        img_output_dir='/var/tmp/matlab/VOC2007/VOCdevkit/VOC2007/JPEGImages',
        anno_output_dir='/var/tmp/matlab/VOC2007/VOCdevkit/VOC2007/Annotations',
        index_file='/var/tmp/matlab/VOC2007/VOCdevkit/VOC2007/ImageSets/Main/traingen.txt'
    )
else: 
    SETTINGS = dict( 
        patches_dir='/Users/slimgee/Desktop/stuff/VOC2007/VOCdevkit/VOC2007/JPEGImages',
        src_dir='/Users/slimgee/git/data/newbike',
        dst_dir='/Users/slimgee/git/data/afew',
        posed_dir='/Users/slimgee/git/data/posed_bikes',
        img_output_dir='/Users/slimgee/git/data/output/images',
        anno_output_dir='/Users/slimgee/git/data/output/annotations',
        index_file='/Users/slimgee/git/data/output/traingen.txt'
    )
