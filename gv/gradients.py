from __future__ import division
import numpy as np
import scipy.signal
from gv.fast import convert_new

def extract(im, orientations=8):
    kern = np.array([[-1, 0, 1]]) / np.sqrt(2)

    #angs = np.linspace(0, 2*np.pi, orientations+1)[:-1]
    #ang_cos = np.cos(angs)
    #ang_sin = np.sin(angs)

    gr_x = scipy.signal.convolve(im, kern, mode='same')
    gr_y = scipy.signal.convolve(im, kern.T, mode='same')

    theta = (orientations - np.round(orientations * (np.arctan2(gr_y, gr_x) + 1.5*np.pi) / (2 * np.pi)).astype(np.int32)) % orientations 
    amps = np.sqrt(gr_x**2 + gr_y**2)

    S = 10
    kern = 1/S * np.ones((S, 1))
    blurred_amps = \
        scipy.signal.convolve(
            scipy.signal.convolve(amps, kern, mode='same'), 
                kern.T, mode='same')

    eps = 0.1 

    amps /= (blurred_amps + eps)

    return convert_new(theta, amps, orientations, 0.01)
