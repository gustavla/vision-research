
import argparse
import sys

parser = argparse.ArgumentParser(description='Superimpose images onto background')
parser.add_argument('fg', metavar='<foreground directory>', help='Directory of foreground images to be superimposed')
parser.add_argument('bg', metavar='<background directory>', help='Directory of background images')
parser.add_argument('dst', metavar='<destination directory>', help='Directory of output images')
parser.add_argument('--size', nargs=2, type=int, default=None, help='Resize source images to this before doing anything else')
parser.add_argument('--crop-size', nargs=2, type=int, default=None, help='Crop to this size after resizing')

args = parser.parse_args()
fg = args.fg
bg = args.bg
dst = args.dst
size = args.size
crop_size = args.crop_size
assert size is not None, "Must specify for now"
assert crop_size is not None, "Must specify for now"
#src = SETTINGS['src_dir']
#dst = SETTINGS['dst_dir']
#img_output_dir = SETTINGS['img_output_dir']
#anno_output_dir = SETTINGS['anno_output_dir']

import gv
import numpy as np
import os.path
import glob
import itertools
import amitgroup as ag
from PIL import Image, ExifTags

#for d, name in [(fg, 'Foreground'), (bg, 'Background'), (dst, 'Destination')]:
    #if not os.path.isdir(d):
        #print "Error: {0} directory must be a directory".format(name)
        #sys.exit(1)

#from PIL import Image

# These have to be png, since we need an alpha channel 
fg_filenames = glob.glob(os.path.join(fg))
bg_filenames = glob.glob(os.path.join(bg))

if not fg_filenames or not bg_filenames:
    print "No files returned"
    sys.exit(1)

print "Foreground images:", len(fg_filenames)
print "Background images:", len(bg_filenames)

#print "Training mixture model"
#from mixture import mixture_from_files
#mixture, images, originals = mixture_from_files(src_filenames, 4)

#lists = mixture.indices_lists()
#mix_comps = mixture.mixture_components()

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
        print ":", fn
        # One images at a time
        im = Image.open(fn)
        im_size = im.size[::-1]
        cur_pos = (0, 0)
        while True:
            cnd_x = cur_pos[0] + size[0] <= im_size[0]
            cnd_y = cur_pos[1] + size[1] <= im_size[1]
            print cur_pos, size, im_size
            print cnd_x, cnd_y
            if not cnd_y:
                # We're done with this image, move on
                break
            elif cnd_x and cnd_y:
                # We have an image
                patch = Image.new("RGBA", size[::-1], (0, 0, 0))
                patch.paste(im, (-cur_pos[1], -cur_pos[0]))
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

#size = 
gen = patch_generator(bg_filenames, crop_size)
print "Going in..."
#idcodes = []
for i, fn in enumerate(fg_filenames):

    print fn
    #fg_im = Image.open(fn)
    fg_im = gv.img.load_image(fn)

    # Resize
    fg_im2 = gv.img.resize(fg_im, size)
        
    # Crop
    fg_im3 = gv.img.crop(fg_im2, crop_size)

    fg_pil = Image.fromarray((fg_im3*255).astype(np.uint8))

    # Now superimpose it onto the dst_im
    try:
        dst_im = gen.next()
    except StopIteration:
        print "Breaking."
        break

    # Find the bounding box in the fg image
    #bnd_box = find_bounding_box(fg_im)
    #print bnd_box

    #index = i + mix_comps[i] * 1000

    #print index
    
    #pos = ((size[0]-fg_im.size[0])//2, (size[1]-fg_im.size[1])//2)
    alpha_composite(fg_pil, dst_im, (0, 0))
    fn = 'sup-{0}'.format(os.path.basename(fn))
    dst_im.save(os.path.join(dst, fn))

    #idcodes.append(idcode)
    #for i, patch in enumerate(patch_generator(dst_filenames, (200, 200))):
        #patch.save('/Users/slimgee/git/data/output/patch{0:02}.jpg'.format(i), 'JPEG')

#open(SETTINGS['index_file'], 'w').write("\n".join(idcodes))
