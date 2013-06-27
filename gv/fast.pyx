#!python:
# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=True
# cython: cdivision=True

import cython
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, abs, fabs, fmax, fmin, log, pow, sqrt
from libc.stdlib cimport rand, srand 

real_p = np.float64
mybool_p = np.uint8
ctypedef np.float64_t real
ctypedef np.uint8_t mybool
#ctypedef cython.floating real

def multifeature_correlate2d_with_mask(np.ndarray[mybool,ndim=3] data_, np.ndarray[real,ndim=3] kernel_, np.ndarray[mybool,ndim=2] mask_):
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

        mybool[:,:,:] data = data_
        real[:,:,:] kernel = kernel_
        real[:,:] response = response_
        mybool[:,:] mask = mask_
    
        real v
        int i, j, sx, sy, f

    for i in range(steps_x):
        for j in range(steps_y):
            if mask[i,j]:
                v = 0
                for sx in range(kernel_d0):
                    for sy in range(kernel_d1):
                        for f in range(num_feat):
                            v += data[i+sx,j+sy,f] * kernel[sx,sy,f]
                response[i,j] = v
            else:
                response[i,j] = -100000

    return response_


def multifeature_correlate2d(np.ndarray[mybool,ndim=3] data_, np.ndarray[real,ndim=3] kernel_):
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

        mybool[:,:,:] data = data_
        real[:,:,:] kernel = kernel_
        real[:,:] response = response_
    
        real v
        int i, j, sx, sy, f

    for i in range(steps_x):
        for j in range(steps_y):
            v = 0
            for sx in range(kernel_d0):
                for sy in range(kernel_d1):
                    for f in range(num_feat):
                        v += data[i+sx,j+sy,f] * kernel[sx,sy,f]
            response[i,j] = v

    return response_

def multifeature_real_correlate2d(np.ndarray[real,ndim=3] data_, np.ndarray[real,ndim=3] kernel_):
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
    
        real v
        int i, j, sx, sy, f

    for i in range(steps_x):
        for j in range(steps_y):
            v = 0
            for sx in range(kernel_d0):
                for sy in range(kernel_d1):
                    for f in range(num_feat):
                        v += data[i+sx,j+sy,f] * kernel[sx,sy,f]
            response[i,j] = v

    return response_



def llh(np.ndarray[mybool,ndim=3] data_, np.ndarray[real,ndim=3] kernel_, np.ndarray[mybool,ndim=2] support_):
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
        #np.ndarray[real,ndim=2] Zs_ = np.zeros((data_d0, data_d1), dtype=real_p)

        mybool[:,:,:] data = data_
        mybool[:,:] support = support_ 
        real[:,:,:] kernel = kernel_
        real[:,:] response = response_
        #real[:,:] Zs = Zs_

        real a = 0.0
        real backprob = 0.0
        real Z = 0.0
        real kxy = 0.0
        real res = 0.0
        int norm = 0
    
        int i, j, sx, sy, f

    for i in range(steps_x):
        for j in range(steps_y):
            Z = 0.0
            res = 0.0
            for f in range(num_feat):
                backprob = 0.0
                norm = 0
                for sx in range(kernel_d0):
                    for sy in range(kernel_d1):
                        if support[sx,sy] == 0:
                            backprob += data[i+sx,j+sy,f]
                            norm += 1
                backprob /= norm
                #backprob /= kernel_d0 * kernel_d1
                if backprob < 0.05:
                    backprob = 0.05

                for sx in range(kernel_d0):
                    for sy in range(kernel_d1):
                        a = log(kernel[sx,sy,f] / (1 - kernel[sx,sy,f]) * (1 - backprob) / backprob)
                        Z += pow(a, 2) * backprob * (1 - backprob)
                        res += (<real>data[i+sx,j+sy,f] - backprob) * a 
    
                # Now normalize this response
            #response[i,j] = res

            # Normalize
            response[i,j] = res / sqrt(Z)

    return response_
