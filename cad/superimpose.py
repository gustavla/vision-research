
#import argparse
import sys

#parser = argparse.ArgumentParser(description='Superimpose images onto background')
#parser.add_argument('src', metavar='<source directory>', help='Directory of foreground images to be superimposed')
#parser.add_argument('dst', metavar='<destination directory>', help='Directory of background images')

#args = parser.parse_args()
from config import SETTINGS
#src = args.src
#dst = args.dst
src = SETTINGS['src_dir']
dst = SETTINGS['dst_dir']
img_output_dir = SETTINGS['img_output_dir']
anno_output_dir = SETTINGS['anno_output_dir']

import numpy as np
import os.path
import glob
import itertools
import amitgroup as ag
from PIL import Image, ExifTags

for d, name in [(src, 'Source'), (dst, 'Destination'), (img_output_dir, 'Image output'), (anno_output_dir, 'Annotation output')]:
    if not os.path.isdir(d):
        print "Error: {0} directory must be a directory".format(name)
        sys.exit(1)

#from PIL import Image

# These have to be png, since we need an alpha channel 
src_filenames = glob.glob(os.path.join(src, '*.png'))
dst_filenames = glob.glob(os.path.join(dst, '*'))

print "Training mixture model"
from mixture import mixture_from_files
mixture, images, originals = mixture_from_files(src_filenames, 4)

#lists = mixture.indices_lists()
mix_comps = mixture.mixture_components()

#templates = mixture.remix(originals)
#ag.plot.images(templates)

# We need more background files than foreground files
# Check if len(src_files) <= len(dst_files) ?

class VOCImage:
    def __init__(self, im, labels=[]):
        self.im = im
        self.labels = labels


if 0:
    def load_exif_corrected_image(filename):
            image=Image.open(filename)
            for orientation in ExifTags.TAGS.keys() : 
                if ExifTags.TAGS[orientation]=='Orientation' : break 
            exif=dict(image._getexif().items())

            if   exif[orientation] == 3 : 
                image=image.rotate(180, expand=True)
            elif exif[orientation] == 6 : 
                image=image.rotate(270, expand=True)
            elif exif[orientation] == 8 : 
                image=image.rotate(90, expand=True)

            return image

def patch_generator(im_filenames, size):
    """Yields a bunch of PIL Image objects of `size`, carved out image files found in the filenames `im_filenames`."""
    
    for fn in im_filenames:
        # One images at a time
        im = Image.open(fn)
        cur_pos = (0, 0)
        while True:
            cnd_x = cur_pos[0] + size[0] < im.size[0]
            cnd_y = cur_pos[1] + size[1] < im.size[1]
            if not cnd_y:
                # We're done with this image, move on
                break
            elif cnd_x and cnd_y:
                # We have an image
                patch = Image.new("RGBA", size, (0, 0, 0))
                patch.paste(im, (-cur_pos[0], -cur_pos[1]))
                #import matplotlib.pylab as plt
                #import pdb; pdb.set_trace()
                yield patch

                # Increment
                cur_pos = (cur_pos[0] + size[0], cur_pos[1]) 
            elif not cnd_x:
                cur_pos = (0, cur_pos[1] + size[1]) 


def alpha_composite(src_im, dst_im, pos):
    #r, g, b, a = src_im.split()
    #src_im = Image.merge("RGB", (r, g, b))
    #mask = Image.merge("L", (a,))
    #dst_im.paste(src_im, (0, 0), mask)
    #dst_im.save("over.png")
    dst_im.paste(src_im, pos, src_im)


def find_bounding_box(im):
    first = True 
    pixels = im.getdata()
    bounding_box = [np.inf, np.inf, -np.inf, -np.inf]
    for x, y in itertools.product(*map(xrange, im.size)):
        p = pixels[y * im.size[0] + x]
        if p[3] > 0:
            bounding_box[0] = min(x, bounding_box[0])
            bounding_box[1] = min(y, bounding_box[1])
            bounding_box[2] = max(x, bounding_box[2])
            bounding_box[3] = max(y, bounding_box[3])
    # Make a bit bigger
    bounding_box = (bounding_box[0]-1, bounding_box[1]-1, bounding_box[2]+1, bounding_box[3]+1)
        
    return tuple(bounding_box)

size = (128, 128)
gen = patch_generator(dst_filenames, size)
xml_template = open(SETTINGS['xml_template']).read()
idcodes = []
for i, fn in enumerate(src_filenames):
    src_im = Image.open(fn)
    # Now superimpose it onto the dst_im
    try:
        dst_im = gen.next()
    except StopIteration:
        break

    # Find the bounding box in the src image
    bnd_box = find_bounding_box(src_im)
    print bnd_box

    index = i + mix_comps[i] * 1000
    
    pos = ((size[0]-src_im.size[0])//2, (size[1]-src_im.size[1])//2)
    alpha_composite(src_im, dst_im, pos)
    idcode = '{0:06}'.format(100000+index)
    fn = '{0}.jpg'.format(idcode)
    dst_im.save(os.path.join(img_output_dir, fn))

    # Also generate an xml with information
    content = xml_template.format(
        filename=fn,
        width=dst_im.size[0], 
        height=dst_im.size[1],
        xmin=pos[0]+bnd_box[0],
        ymin=pos[1]+bnd_box[1],
        xmax=pos[0]+bnd_box[2],
        ymax=pos[1]+bnd_box[3]
    )
     
    open(os.path.join(anno_output_dir, '{0}.xml'.format(idcode)), 'w').write(content)
    
    idcodes.append(idcode)
    #for i, patch in enumerate(patch_generator(dst_filenames, (200, 200))):
        #patch.save('/Users/slimgee/git/data/output/patch{0:02}.jpg'.format(i), 'JPEG')

open(SETTINGS['index_file'], 'w').write("\n".join(idcodes))
