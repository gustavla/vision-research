
import amitgroup as ag
import gv
from config import VOCSETTINGS

images = gv.voc.load_object_images_of_size(VOCSETTINGS, 'bicycle', (128, 128), dataset='train')

ag.plot.images(images[:9])
