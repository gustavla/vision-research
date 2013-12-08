from __future__ import division
import numpy as np
import scipy.signal
import amitgroup as ag
from gv.fast import convert_new

def extract(im, orientations=8, threshold=0.000001, eps=0.0001, blur_size=10):
    kern = np.array([[-1, 0, 1]]) / np.sqrt(2)
    gr_x = scipy.signal.convolve(im, kern, mode='same')
    gr_y = scipy.signal.convolve(im, kern.T, mode='same')

    theta = (orientations - np.round(orientations * (np.arctan2(gr_y, gr_x) + 1.5*np.pi) / (2 * np.pi)).astype(np.int32)) % orientations 
    amps = np.sqrt(gr_x**2 + gr_y**2)

    S = blur_size 
    cumsum_amps = amps.cumsum(0).cumsum(1) / S**2
    hS = S // 2
    padded = ag.util.border_value_pad(cumsum_amps, hS+1)
    dim0, dim1 = im.shape[:2]
    blurred_amps = padded[S:S+dim0,S:S+dim1] - padded[0:0+dim0,S:S+dim1] - padded[S:S+dim0,0:0+dim1] + padded[0:0+dim0,0:0+dim1]

    amps /= (blurred_amps + eps)
    return convert_new(theta, amps, orientations, threshold)
