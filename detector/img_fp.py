
import amitgroup as ag
import gv
from config import VOCSETTINGS

images = gv.voc.load_negative_images_of_size(VOCSETTINGS, 'bicycle', (128, 128), dataset='train', count=9, padding=2)

ag.plot.images(images[:9])
