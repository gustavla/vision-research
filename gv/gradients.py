from __future__ import division
import numpy as np
import scipy.signal
import amitgroup as ag
from gv.fast import convert_new, convert_new_float_TEMP

def box_blur(im, S):
    cumsum_im = im.cumsum(0).cumsum(1) / S**2
    hS = S // 2
    padded = ag.util.border_value_pad(cumsum_im, hS+1)
    dim0, dim1 = im.shape[:2]
    blurred_im = padded[S:S+dim0,S:S+dim1] - padded[0:0+dim0,S:S+dim1] - padded[S:S+dim0,0:0+dim1] + padded[0:0+dim0,0:0+dim1]
    return blurred_im

def orientations(im, orientations=8):
    kern = np.array([[-1, 0, 1]]) / np.sqrt(2)
    gr_x = scipy.signal.convolve(im, kern, mode='same')
    gr_y = scipy.signal.convolve(im, kern.T, mode='same')

    #theta = orientations - orientations * ((np.arctan2(gr_y, gr_x) + 1.5*np.pi) / (2 * np.pi)) % orientations 
    theta = np.arctan2(gr_y, gr_x)
    amps = np.sqrt(gr_x**2 + gr_y**2)

    w = np.min(theta.shape) // 4
    print theta.shape, w
    theta = theta[theta.shape[0]//2-w:theta.shape[0]//2+w, theta.shape[1]//2-w:theta.shape[1]//2+w]
    amps = amps[theta.shape[0]//2-w:theta.shape[0]//2+w, theta.shape[1]//2-w:theta.shape[1]//2+w]

    theta = theta.ravel()
    gr_x = gr_x.ravel()
    gr_y = gr_y.ravel()
    amps = amps.ravel()

    #II = amps > 0.025
    #theta = theta[II]
    #gr_x = gr_x[II]
    #gr_y = gr_y[II]

    return theta, gr_x, gr_y

def extract(im, orientations=8, threshold=0.000001, eps=0.01, blur_size=10):
    #threshold=0.1
    #eps=0.01
    #blur_size=30

    kern = np.array([[-1, 0, 1]]) / np.sqrt(2)

    # This is slow, below is equivalent
    #gr_x = scipy.signal.convolve(im, kern, mode='same')
    #gr_y = scipy.signal.convolve(im, kern.T, mode='same')

    im_padded = ag.util.zeropad(im, 1)
    gr_x = (im_padded[1:-1,:-2] - im_padded[1:-1,2:]) / np.sqrt(2)
    gr_y = (im_padded[:-2,1:-1] - im_padded[2:,1:-1]) / np.sqrt(2)

    theta = (orientations - np.round(orientations * (np.arctan2(gr_y, gr_x) + 1.5*np.pi) / (2 * np.pi)).astype(np.int32)) % orientations 
    amps = np.sqrt(gr_x**2 + gr_y**2)

    blurred_amps = box_blur(amps, blur_size)
    amps /= (blurred_amps + eps)

    #center_amps = box_blur(amps, 2)
    return convert_new(theta, amps, orientations, threshold)
    #return convert_new_float_TEMP(theta, amps, orientations, threshold), amps 
