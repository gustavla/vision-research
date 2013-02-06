#!python:
# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=True
# cython: cdivision=True

import cython
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, abs, fabs, fmax, fmin, log
from libc.stdlib cimport rand, srand 

real_p = np.float64
ctypedef np.float64_t real
#ctypedef cython.floating real

def masked_convolve(np.ndarray[real,ndim=3] data_, np.ndarray[real,ndim=3] kernel_):
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
        real[:,:] response = response_
    
        int i, j, sx, sy, f

    for i in range(steps_x):
        for j in range(steps_y):
            for sx in range(kernel_d0):
                for sy in range(kernel_d1):
                    for f in range(num_feat):
                        response[i,j] += data[i+sx,j+sy,f] * kernel[sx,sy,f]

    return response_



def llh(np.ndarray[real,ndim=3] data_, np.ndarray[real,ndim=3] kernel_):
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
        np.ndarray[real,ndim=2] response_ = np.zeros((steps_x, steps_y), dtype=real_p)

        real[:,:,:] data = data_
        real[:,:,:] kernel = kernel_
        real[:,:] response = response_

        real backprob = 0.0
    
        int i, j, sx, sy, f

    for i in range(steps_x):
        for j in range(steps_y):
            for f in range(num_feat):
                backprob = 0.0
                for sx in range(kernel_d0):
                    for sy in range(kernel_d1):
                        backprob += data[i+sx,j+sy,f]
                backprob /= kernel_d0 * kernel_d1
                if backprob < 0.05:
                    backprob = 0.05

                for sx in range(kernel_d0):
                    for sy in range(kernel_d1):
                            response[i,j] += data[i+sx,j+sy,f] * log(kernel[sx,sy,f] / (1 - kernel[sx,sy,f]) * (1 - backprob) / backprob) 

    return response_
