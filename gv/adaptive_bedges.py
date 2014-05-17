from __future__ import division, print_function, absolute_import
import numpy as np
import amitgroup as ag

def adaptive_bedges(images, k=6, spread='box', radius=1, minimum_contrast_multiple=0.0, minimum_contrast=None, contrast_insensitive=False, first_axis=False, preserve_size=True, pre_blurring=None, blur_size=1):
    single = len(images.shape) == 2
    if single:
        images = images[np.newaxis]

    from gv.fast import adaptive_array_bedges
    from gv.gradients import box_blur

    kern = np.array([[-1, 0, 1]]) / np.sqrt(2)

    im_padded = ag.util.zeropad(images, (0, 1, 1))
    gr_x = (im_padded[1:-1,:-2] - im_padded[1:-1,2:]) / np.sqrt(2)
    gr_y = (im_padded[:-2,1:-1] - im_padded[2:,1:-1]) / np.sqrt(2)

    #theta = (orientations - np.round(orientations * (np.arctan2(gr_y, gr_x) + 1.5*np.pi) / (2 * np.pi)).astype(np.int32)) % orientations 
    amps = np.sqrt(gr_x**2 + gr_y**2)

    blurred_amps = box_blur(amps, blur_size)

    features = adaptive_array_bedges(images, amps, k, minimum_contrast_multiple, contrast_insensitive) 

    # Spread the feature
    features = ag.features.bspread(features, radius=radius, spread=spread, first_axis=True)

    # Skip the 2-pixel border that is not valid
    if not preserve_size:
        features = features[:,:,2:-2,2:-2] 

    if not first_axis:
        features = np.rollaxis(features, axis=1, start=features.ndim)
            
    if single:
        features = features[0]

    return features
