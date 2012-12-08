#!python:
# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=True
# cython: cdivision=True

import cython
import numpy as np
cimport numpy as np
from libc.math cimport exp, abs, fabs, fmax, fmin
from libc.stdlib cimport rand, srand 

real_p = np.float64
ctypedef np.float64_t real

def masked_convolve(data_, kernel_, kernel_mask_):
    assert data_.shape[0] > kernel_.shape[0]
    assert data_.shape[1] > kernel_.shape[1]
    cdef:
        int data_d0 = data_.shape[0]
        int data_d1 = data_.shape[1]
        int kernel_d0 = kernel_.shape[0]
        int kernel_d1 = kernel_.shape[1]
        int steps_x = (data_d0 - kernel_d0) + 1
        int steps_y = (data_d1 - kernel_d1) + 1
        int num_feat = data_.shape[2]

        #int size_d0 = min(data_d0, kernel_d0)
        #int size_d1 = min(data_d1, kernel_d1)
        np.ndarray[real,ndim=2] response_ = np.zeros((steps_x, steps_y))

        real[:,:,:] data = data_
        real[:,:,:] kernel = kernel_
        np.int8_t[:,:] kernel_mask = kernel_mask_
        real[:,:] response = response_

    print 'data:', data_.shape
    print 'kernel:', kernel_.shape
    print 'num_feat', num_feat

    for i in range(steps_x):
        for j in range(steps_y):
            for sx in range(kernel_d0):
                for sy in range(kernel_d1):
                    #if kernel_mask[sx,sy] == 1:
                    for f in range(num_feat):
                        response[i,j] += data[i+sx,j+sy,f] * kernel[sx,sy,f]

    return response_
