
import amitgroup as ag
import gv

images = gv.voc.load_negative_images_of_size('bicycle', (128, 128), dataset='train', count=9, padding=2)

ag.plot.images(images[:9])
