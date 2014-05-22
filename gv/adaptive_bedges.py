from __future__ import division, print_function, absolute_import
import numpy as np
import amitgroup as ag

def box_blur_multi(im, S):
    cumsum_im = im.cumsum(1).cumsum(2) / S**2
    hS = S // 2
    padded = ag.util.border_value_pad(cumsum_im, (0, hS+1, hS+1))
    dim0, dim1 = im.shape[1:3]
    blurred_im = padded[:,S:S+dim0,S:S+dim1] - padded[:,0:0+dim0,S:S+dim1] - padded[:,S:S+dim0,0:0+dim1] + padded[:,0:0+dim0,0:0+dim1]
    return blurred_im

def adaptive_bedges(images, k=6, spread='box', radius=1, minimum_contrast_multiple=0.0, minimum_contrast=None, contrast_insensitive=False, first_axis=False, preserve_size=True, pre_blurring=None, blur_size=1):
    single = images.ndim == 2
    if single:
        images = images[np.newaxis]

    from gv.fast import adaptive_array_bedges

    kern = np.array([[-1, 0, 1]]) / np.sqrt(2)

    im_padded = ag.util.zeropad(images, (0, 1, 1))
    gr_x = (im_padded[:,1:-1,:-2] - im_padded[:,1:-1,2:]) / np.sqrt(2)
    gr_y = (im_padded[:,:-2,1:-1] - im_padded[:,2:,1:-1]) / np.sqrt(2)

    #theta = (orientations - np.round(orientations * (np.arctan2(gr_y, gr_x) + 1.5*np.pi) / (2 * np.pi)).astype(np.int32)) % orientations 
    amps = np.sqrt(gr_x**2 + gr_y**2)

    #blurred_amps = box_blur_multi(amps, blur_size)
    import scipy.ndimage.filters

    blurred_amps = scipy.ndimage.filters.percentile_filter(amps, minimum_contrast_multiple, (1, blur_size, blur_size))

    features = adaptive_array_bedges(images, blurred_amps, k, contrast_insensitive) 

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
