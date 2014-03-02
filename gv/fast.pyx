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

cdef real sigmoid(real x) nogil:
    return 1.0 / (1.0 + exp(-x))

cdef real logit(real x) nogil:
    return log(x / (1 - x))

def multifeature_correlate2d_with_mask(np.ndarray[mybool,ndim=3] data_, np.ndarray[real,ndim=3] kernel_, np.ndarray[mybool,ndim=2] mask_):
    assert data_.shape[0] >= kernel_.shape[0]
    assert data_.shape[1] >= kernel_.shape[1]
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


def multifeature_correlate2d(np.ndarray[mybool,ndim=3] data_, np.ndarray[real,ndim=3] kernel_, strides=(1, 1)):
    assert data_.shape[0] >= kernel_.shape[0]
    assert data_.shape[1] >= kernel_.shape[1]
    cdef:
        int data_d0 = data_.shape[0]
        int data_d1 = data_.shape[1]
        int kernel_d0 = kernel_.shape[0]
        int kernel_d1 = kernel_.shape[1]
        int stride0 = <int>strides[0]
        int stride1 = <int>strides[1]
        int steps_x = (data_d0 - kernel_d0) // stride0 + 1
        int steps_y = (data_d1 - kernel_d1) // stride1 + 1
        int num_feat = data_.shape[2]

        #int size_d0 = min(data_d0, kernel_d0)
        #int size_d1 = min(data_d1, kernel_d1)
        np.ndarray[real,ndim=2] response_ = np.zeros((steps_x, steps_y))

        mybool[:,:,:] data = data_
        real[:,:,:] kernel = kernel_
        real[:,:] response = response_
    
        real v
        int i, j, sx, sy, f, ii, jj

    with nogil:
        for i in range(steps_x):
            ii = i * stride0
            for j in range(steps_y):
                jj = j * stride1
                v = 0
                for sx in range(kernel_d0):
                    for sy in range(kernel_d1):
                        for f in range(num_feat):
                            v += data[ii+sx,jj+sy,f] * kernel[sx,sy,f]
                response[i,j] = v

    return response_

def multifeature_correlate2d_with_indices(np.ndarray[mybool,ndim=3] data_, 
                                          np.ndarray[real,ndim=3] kernel_, 
                                          np.ndarray[np.int32_t, ndim=2] indices_, 
                                          strides=(1, 1)):
    assert data_.shape[0] >= kernel_.shape[0]
    assert data_.shape[1] >= kernel_.shape[1]
    assert indices_.shape[1] == 3
    cdef:
        int data_d0 = data_.shape[0]
        int data_d1 = data_.shape[1]
        int kernel_d0 = kernel_.shape[0]
        int kernel_d1 = kernel_.shape[1]
        int stride0 = <int>strides[0]
        int stride1 = <int>strides[1]
        int steps_x = (data_d0 - kernel_d0) // stride0 + 1
        int steps_y = (data_d1 - kernel_d1) // stride1 + 1
        int num_feat = data_.shape[2]
        int num_indices = indices_.shape[0]

        #int size_d0 = min(data_d0, kernel_d0)
        #int size_d1 = min(data_d1, kernel_d1)
        np.ndarray[real,ndim=2] response_ = np.zeros((steps_x, steps_y))

        mybool[:,:,:] data = data_
        real[:,:,:] kernel = kernel_
        np.int32_t[:,:] indices = indices_
        real[:,:] response = response_
    
        real v
        int i, j, ind, sx, sy, f, ii, jj

    #with nogil:
    #    for i in range(steps_x):
    #        for j in range(steps_y):
    #            v = 0
    #            for ind in range(num_indices):
    #                sx = indices[ind,0]     
    #                sy = indices[ind,1]
    #                f = indices[ind,2]
    #                v += data[i+sx,j+sy,f] * kernel[sx,sy,f]
    #                #for f in range(num_feat):
    #                    #v += data[i+sx,j+sy,f] * kernel[sx,sy,f]
    #            response[i,j] = v

    with nogil:
        for ind in range(num_indices):
            sx = indices[ind,0]     
            sy = indices[ind,1]
            f = indices[ind,2]
            for i in range(steps_x):
                ii = i * stride0
                for j in range(steps_y):
                    jj = j * stride1
                    #v = 0
                    v = data[ii+sx,jj+sy,f] * kernel[sx,sy,f]
                    #for f in range(num_feat):
                        #v += data[i+sx,j+sy,f] * kernel[sx,sy,f]
                    response[i,j] += v

    return response_

def multifeature_correlate2d_multi(np.ndarray[mybool,ndim=3] data_, np.ndarray[real,ndim=4] kernels_, np.ndarray[np.int32_t,ndim=2] bkgcomps_):
    assert data_.shape[0] >= kernels_.shape[1]
    assert data_.shape[1] >= kernels_.shape[2]
    cdef:
        int data_d0 = data_.shape[0]
        int data_d1 = data_.shape[1]
        int kernel_d0 = kernels_.shape[1]
        int kernel_d1 = kernels_.shape[2]
        int steps_x = (data_d0 - kernel_d0) + 1
        int steps_y = (data_d1 - kernel_d1) + 1
        int num_feat = data_.shape[2]

        #int size_d0 = min(data_d0, kernel_d0)
        #int size_d1 = min(data_d1, kernel_d1)
        np.ndarray[real,ndim=2] response_ = np.zeros((steps_x, steps_y))

        mybool[:,:,:] data = data_
        real[:,:,:,:] kernels = kernels_
        np.int32_t[:,:] bkgcomps = bkgcomps_
        real[:,:] response = response_
    
        real v
        int i, j, sx, sy, f
        np.int32_t comp

    with nogil:
        for i in range(steps_x):
            for j in range(steps_y):
                v = 0
                comp = bkgcomps[i,j]
                for sx in range(kernel_d0):
                    for sy in range(kernel_d1):
                        for f in range(num_feat):
                            v += data[i+sx,j+sy,f] * kernels[comp,sx,sy,f]
                response[i,j] = v

    return response_


def multifeature_real_correlate2d(np.ndarray[real,ndim=3] data_, np.ndarray[real,ndim=3] kernel_, strides=(1, 1)):
    assert data_.shape[0] >= kernel_.shape[0]
    assert data_.shape[1] >= kernel_.shape[1]
    cdef:
        int data_d0 = data_.shape[0]
        int data_d1 = data_.shape[1]
        int kernel_d0 = kernel_.shape[0]
        int kernel_d1 = kernel_.shape[1]
        int stride0 = <int>strides[0]
        int stride1 = <int>strides[1]
        int steps_x = (data_d0 - kernel_d0) // stride0 + 1
        int steps_y = (data_d1 - kernel_d1) // stride1 + 1
        int num_feat = data_.shape[2]

        #int size_d0 = min(data_d0, kernel_d0)
        #int size_d1 = min(data_d1, kernel_d1)
        np.ndarray[real,ndim=2] response_ = np.zeros((steps_x, steps_y))

        real[:,:,:] data = data_
        real[:,:,:] kernel = kernel_
        real[:,:] response = response_
    
        real v
        int i, j, sx, sy, f, ii, jj

    for i in range(steps_x):
        ii = i * stride0
        for j in range(steps_y):
            jj = j * stride1
            v = 0
            for sx in range(kernel_d0):
                for sy in range(kernel_d1):
                    for f in range(num_feat):
                        v += data[ii+sx,jj+sy,f] * kernel[sx,sy,f]
            response[i,j] = v

    return response_



def llh(np.ndarray[mybool,ndim=3] data_, np.ndarray[real,ndim=3] kernel_, np.ndarray[mybool,ndim=2] support_):
    assert data_.shape[0] >= kernel_.shape[0]
    assert data_.shape[1] >= kernel_.shape[1]
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

def nonparametric_rescore(np.ndarray[real,ndim=2] res, start, step, np.ndarray[real,ndim=1] points):
    cdef:
        int N = len(points)
        real[:,:] res_mv = res
        real real_start = <real>start
        real real_step = <real>step
        real real_end = <real>(real_start + real_step * (N-1))
        int dim0 = res.shape[0]
        int dim1 = res.shape[1]
        int i, j, index
        real s
        real pre_slope = (points[1] - points[0]) / step
        real post_slope = (points[N-1] - points[N-2]) / step
        real[:] points_mv = points 

    with nogil:
        for i in range(dim0):
            for j in range(dim1):
                s = res_mv[i,j]
                if s < real_start:
                    s = points[0] + (s - real_start) * pre_slope
                else:
                    index = <int>((s - real_start) // real_step)
                    if index >= N-1:
                        s = points[N-1] + (s - real_end) * post_slope
                    else:
                        s = points[index] + ((s - real_start) / real_step - index) * (points[index+1] - points[index])

                res_mv[i,j] = s

def bkg_model_dists(np.ndarray[mybool,ndim=3] feats, np.ndarray[real,ndim=2] bkgs, size, padding=0, inner_padding=-1000):
    assert feats.shape[2] == bkgs.shape[1]
    assert padding > inner_padding
    cdef:
        int size0 = <int>size[0]
        int size1 = <int>size[1]
        int ip = <int>padding
        int iip = <int>inner_padding
        int num_bkgs = bkgs.shape[0]
        int dim0 = feats.shape[0] - size0 + 1
        int dim1 = feats.shape[1] - size1 + 1
        int dim2 = feats.shape[2]

        int if_dim0 = feats.shape[0] + 1 + 2*ip
        int if_dim1 = feats.shape[1] + 1 + 2*ip
        np.ndarray[real,ndim=3] integral_feats = np.zeros((if_dim0, if_dim1, feats.shape[2]))
        np.ndarray[real,ndim=3] dists = np.zeros((dim0, dim1, num_bkgs))

        real[:,:,:] int_mv = integral_feats
        real[:,:,:] dists_mv = dists
        real[:,:] bkgs_mv = bkgs

        real v, w, s
        np.int32_t count 
        int i, j, f, b

        real inner_area = max(size0 + 2*iip, 0) * max(size1 + 2*iip, 0)
        real outer_area = max(size0 + 2*ip, 0) * max(size1 + 2*ip, 0)

    integral_feats[ip+1:if_dim0-ip,ip+1:if_dim1-ip] = feats.astype(np.int32).cumsum(0).cumsum(1).astype(real_p) / (outer_area - inner_area)

    with nogil:
        # Fill in to the right
        for i in range(if_dim0-ip, if_dim0):
            for j in xrange(if_dim1):
                int_mv[i,j] = int_mv[if_dim0-ip-1,j]

        for i in range(if_dim0):
            for j in xrange(if_dim1-ip, if_dim1):
                int_mv[i,j] = int_mv[i,if_dim1-ip-1]

        for i in range(dim0):
            for j in range(dim1):
                v = 0
                for f in range(dim2): 
                    # Get background value here
                    if inner_area > 0:
                        s = int_mv[i+size0+2*ip,j+size1+2*ip,f] - \
                            int_mv[i           ,j+size1+2*ip,f] - \
                            int_mv[i+size0+2*ip,j           ,f] + \
                            int_mv[i           ,j           ,f] - \
                           (int_mv[ip+i+size0+iip,ip+j+size1+iip,f] - \
                            int_mv[ip+i-iip      ,ip+j+size1+iip,f] - \
                            int_mv[ip+i+size0+iip,ip+j-iip      ,f] + \
                            int_mv[ip+i-iip      ,ip+j-iip      ,f])
                    else:
                        s = int_mv[i+size0+2*ip,j+size1+2*ip,f] - \
                            int_mv[i           ,j+size1+2*ip,f] - \
                            int_mv[i+size0+2*ip,j           ,f] + \
                            int_mv[i           ,j           ,f]

                    for b in range(num_bkgs):
                        w = (bkgs[b,f] - s)
                        dists_mv[i,j,b] += w * w

    return dists

def bkg_model_dists2(np.ndarray[mybool,ndim=3] feats, np.ndarray[real,ndim=2] bkgs, size, L, padding=0):
    assert feats.shape[2] == bkgs.shape[1]
    cdef:
        int size0 = <int>size[0]
        int size1 = <int>size[1]
        int ip = <int>padding
        int iL = <int>L
        int num_bkgs = bkgs.shape[0]
        int dim0 = feats.shape[0] - size0 + 1
        int dim1 = feats.shape[1] - size1 + 1
        int dim2 = feats.shape[2]

        int if_dim0 = feats.shape[0] + 1 + 2*ip
        int if_dim1 = feats.shape[1] + 1 + 2*ip
        np.ndarray[real,ndim=3] integral_feats = np.zeros((if_dim0, if_dim1, feats.shape[2]))
        np.ndarray[real,ndim=3] dists = np.zeros((dim0, dim1, num_bkgs))

        real[:,:,:] int_mv = integral_feats
        real[:,:,:] dists_mv = dists
        real[:,:] bkgs_mv = bkgs

        real v, w, s, bbf
        np.int32_t count 
        int i, j, f, b

    integral_feats[ip+1:if_dim0-ip,ip+1:if_dim1-ip] = feats.astype(np.int32).cumsum(0).cumsum(1).astype(real_p) / ((size0 + 2*ip) * (size1 + 2*ip))

    with nogil:
        # Fill in to the right
        for i in range(if_dim0-ip, if_dim0):
            for j in xrange(if_dim1):
                int_mv[i,j] = int_mv[if_dim0-ip-1,j]

        for i in range(if_dim0):
            for j in xrange(if_dim1-ip, if_dim1):
                int_mv[i,j] = int_mv[i,if_dim1-ip-1]

        for i in range(dim0):
            for j in range(dim1):
                v = 0
                for f in range(dim2): 
                    # Get background value here
                    s = int_mv[i+size0+2*ip,j+size1+2*ip,f] - \
                        int_mv[i           ,j+size1+2*ip,f] - \
                        int_mv[i+size0+2*ip,j           ,f] + \
                        int_mv[i           ,j           ,f]

                    for b in range(num_bkgs):
                        bbf = bkgs_mv[b,f]
                        w = -(s - bbf) * (s - bbf) / (2 * bbf * (1 - bbf) / iL) - 0.5 * log(bbf * (1 - bbf) / iL)
                        dists_mv[i,j,b] += w

    return dists

def bkg_model_dists3(np.ndarray[mybool,ndim=3] feats, np.ndarray[real,ndim=2] bkgs, size, L, padding=1):
    assert feats.shape[2] == bkgs.shape[1]
    cdef:
        int size0 = <int>size[0]
        int size1 = <int>size[1]
        int ip = <int>padding
        int iL = <int>L
        int num_bkgs = bkgs.shape[0]
        int dim0 = feats.shape[0] - size0 + 1
        int dim1 = feats.shape[1] - size1 + 1
        int dim2 = feats.shape[2]

        int if_dim0 = feats.shape[0] + 1 + 2*ip
        int if_dim1 = feats.shape[1] + 1 + 2*ip
        np.ndarray[real,ndim=3] integral_feats = np.zeros((if_dim0, if_dim1, feats.shape[2]))
        np.ndarray[real,ndim=3] dists = np.zeros((dim0, dim1, num_bkgs))

        real[:,:,:] int_mv = integral_feats
        real[:,:,:] dists_mv = dists
        real[:,:] bkgs_mv = bkgs

        real v, w, s, bbf
        np.int32_t count 
        int i, j, f, b

    integral_feats[ip+1:if_dim0-ip,ip+1:if_dim1-ip] = feats.astype(np.int32).cumsum(0).cumsum(1).astype(real_p) / (((size0 + 2*ip) * (size1 + 2*ip)) - size0*size1)

    with nogil:
        # Fill in to the right
        for i in range(if_dim0-ip, if_dim0):
            for j in xrange(if_dim1):
                int_mv[i,j] = int_mv[if_dim0-ip-1,j]

        for i in range(if_dim0):
            for j in xrange(if_dim1-ip, if_dim1):
                int_mv[i,j] = int_mv[i,if_dim1-ip-1]

        for i in range(dim0):
            for j in range(dim1):
                v = 0
                for f in range(dim2): 
                    # Get background value here
                    s = int_mv[i+size0+2*ip,j+size1+2*ip,f] - \
                        int_mv[i           ,j+size1+2*ip,f] - \
                        int_mv[i+size0+2*ip,j           ,f] + \
                        int_mv[i           ,j           ,f]

                    s-= int_mv[ip+i+size0,ip+j+size1,f] - \
                        int_mv[ip+i      ,ip+j+size1,f] - \
                        int_mv[ip+i+size0,ip+j      ,f] + \
                        int_mv[ip+i      ,ip+j      ,f]

                    for b in range(num_bkgs):
                        bbf = bkgs_mv[b,f]
                        w = -(s - bbf) * (s - bbf) / (2 * bbf * (1 - bbf) / iL) - 0.5 * log(bbf * (1 - bbf) / iL)
                        dists_mv[i,j,b] += w

    return dists

def bkg_beta_dists(np.ndarray[mybool,ndim=3] feats, np.ndarray[real,ndim=3] mixture_params, size, padding=0, inner_padding=-1000):
    assert feats.shape[2] == mixture_params.shape[1]
    cdef:
        int size0 = <int>size[0]
        int size1 = <int>size[1]
        int ip = <int>padding
        int iip = <int>inner_padding
        int num_bkgs = mixture_params.shape[0]
        int dim0 = feats.shape[0] - size0 + 1
        int dim1 = feats.shape[1] - size1 + 1
        int dim2 = feats.shape[2]

        int if_dim0 = feats.shape[0] + 1 + 2*ip
        int if_dim1 = feats.shape[1] + 1 + 2*ip
        np.ndarray[real,ndim=3] integral_feats = np.zeros((if_dim0, if_dim1, feats.shape[2]))
        np.ndarray[real,ndim=3] dists = np.zeros((dim0, dim1, num_bkgs))

        real[:,:,:] int_mv = integral_feats
        real[:,:,:] dists_mv = dists
        real[:,:,:] params_mv = mixture_params

        real v, w
        np.ndarray[real,ndim=1] s = np.zeros(dim2)
        np.int32_t count 
        int i, j, f, b

        real area = 0.0

        real inner_area = max(size0 + 2*iip, 0) * max(size1 + 2*iip, 0)
        real outer_area = max(size0 + 2*ip, 0) * max(size1 + 2*ip, 0)

    integral_feats[ip+1:if_dim0-ip,ip+1:if_dim1-ip] = feats.astype(np.int32).cumsum(0).cumsum(1).astype(real_p) / (outer_area - inner_area)

    from scipy.stats import beta

    # Fill in to the right
    for i in range(if_dim0-ip, if_dim0):
        for j in xrange(if_dim1):
            int_mv[i,j] = int_mv[if_dim0-ip-1,j]

    for i in range(if_dim0):
        for j in xrange(if_dim1-ip, if_dim1):
            int_mv[i,j] = int_mv[i,if_dim1-ip-1]

#    for i in range(dim0):
#        for j in range(dim1):
#            v = 0
#            for f in range(dim2): 
#                # Get background value here
#                s = int_mv[i+size0+2*ip,j+size1+2*ip,f] - \
#                    int_mv[i           ,j+size1+2*ip,f] - \
#                    int_mv[i+size0+2*ip,j           ,f] + \
#                    int_mv[i           ,j           ,f]
#
#                for b in range(num_bkgs):
#                    # TODO: Return as dist or score? Probably score.
#                    dists_mv[i,j,b] -= beta.logpdf(s, params_mv[b,f,0], params_mv[b,f,1])

    for i in range(dim0):
        for j in range(dim1):
            #v = 0
            #for f in range(dim2): 
            # Get background value here
            if inner_area > 0:
                s[:] = integral_feats[i+size0+2*ip,j+size1+2*ip] - \
                    integral_feats[i           ,j+size1+2*ip] - \
                    integral_feats[i+size0+2*ip,j           ] + \
                    integral_feats[i           ,j           ] - \
                   (integral_feats[ip+i+size0+iip,ip+j+size1+iip] - \
                    integral_feats[ip+i-iip      ,ip+j+size1+iip] - \
                    integral_feats[ip+i+size0+iip,ip+j-iip      ] + \
                    integral_feats[ip+i-iip      ,ip+j-iip      ])
            else:
                s[:] = integral_feats[i+size0+2*ip,j+size1+2*ip] - \
                    integral_feats[i           ,j+size1+2*ip] - \
                    integral_feats[i+size0+2*ip,j           ] + \
                    integral_feats[i           ,j           ]


            #s[:] = np.clip(s, 0.01, 1-0.01)
            np.clip(s, 0.01, 1-0.01, out=s)

            dists[i,j] = -np.sum(beta.logpdf(s, mixture_params[...,0], mixture_params[...,1]), axis=1) 
            #dists[i,j] = -np.sum(np.clip(beta.logpdf(s, mixture_params[...,0], mixture_params[...,1]), -10, 10), axis=1) 
            #dists[i,j] = -np.sum(np.log((1-1e-10)*beta.pdf(s, mixture_params[...,0], mixture_params[...,1]) + 1e-10), axis=1) 
            #for b in xrange(num_bkgs):
                #dists_mv[i,j,b] -= np.sum(beta.logpdf(s, mixture_params[b,:,0], mixture_params[b,:,1]))
                #for b in range(num_bkgs):
                    # TODO: Return as dist or score? Probably score.
                    #dists_mv[i,j,b] -= beta.logpdf(s, params_mv[b,f,0], params_mv[b,f,1])

    return dists

def correlate_abunch(np.ndarray[np.uint8_t,ndim=1] X, np.ndarray[np.uint8_t,ndim=4] Y):
    cdef:
        np.ndarray[real,ndim=3] ret = np.empty((Y.shape[1], Y.shape[2], Y.shape[3]))

        real X_std = np.std(X, ddof=1)
        np.ndarray[real,ndim=3] Y_stds = Y.std(axis=0, ddof=1)

        real[:,:,:] ret_mv = ret
        real[:,:,:] Y_stds_mv = Y_stds
        real X_mean = X.mean()
        np.uint8_t[:] X_mv = X
        np.uint8_t[:,:,:,:] Y_mv = Y
        int i, j, k, n
        int N = Y.shape[0]
        int dim0 = Y.shape[1]
        int dim1 = Y.shape[2]
        int dim2 = Y.shape[3]
        real corr = 0.0

    if 1:
        for i in range(dim0):
            for j in range(dim1):
                for k in range(dim2):
                    corr = 0.0
                    if X_std == 0.0 or Y_stds_mv[i,j,k] == 0.0:
                        ret_mv[i,j,k] = 0.0
                    else:
                        for n in range(N):
                            corr += (X_mv[n] - X_mean) * Y_mv[n,i,j,k]

                        ret_mv[i,j,k] = corr / (<real>(N - 1) * X_std * Y_stds_mv[i,j,k])

    return ret

def convert_new(np.ndarray[dtype=np.int32_t,ndim=2] theta, np.ndarray[dtype=np.float64_t,ndim=2] amplitudes, int num_orientations, double threshold):
    cdef int dim0 = theta.shape[0]
    cdef int dim1 = theta.shape[1]
    cdef double a
    cdef int i, f, v
    cdef np.ndarray[dtype=np.uint8_t, ndim=3] feats = np.zeros((dim0, dim1, num_orientations), dtype=np.uint8)
    cdef np.uint8_t[:,:,:] feats_mv = feats

    cdef np.int32_t[:,:] theta_mv = theta 
    cdef np.float64_t[:,:] amplitudes_mv = amplitudes

    with nogil:
        for i in range(dim0):
            for j in range(dim1):
                a = amplitudes_mv[i,j]
                if a >= threshold:
                    v = theta_mv[i,j] 
                    feats_mv[i,j,v] = 1

    return feats

def convert_new_float_TEMP(np.ndarray[dtype=np.int32_t,ndim=2] theta, np.ndarray[dtype=np.float64_t,ndim=2] amplitudes, int num_orientations, double threshold):
    cdef int dim0 = theta.shape[0]
    cdef int dim1 = theta.shape[1]
    cdef double a
    cdef int i, f, v
    cdef np.ndarray[dtype=np.float64_t, ndim=3] feats = np.zeros((dim0, dim1, num_orientations), dtype=np.float64)
    cdef np.float64_t[:,:,:] feats_mv = feats

    cdef np.int32_t[:,:] theta_mv = theta 
    cdef np.float64_t[:,:] amplitudes_mv = amplitudes

    with nogil:
        for i in range(dim0):
            for j in range(dim1):
                a = amplitudes_mv[i,j]
                if a >= threshold:
                    v = theta_mv[i,j] 
                    feats_mv[i,j,v] = a 

    return feats

def resample_and_arrange_image(np.ndarray[dtype=np.uint8_t,ndim=2] image, target_size, np.ndarray[dtype=np.float64_t,ndim=2] lut):
    cdef:
        int dim0 = image.shape[0]
        int dim1 = image.shape[1]
        int output_dim0 = target_size[0]
        int output_dim1 = target_size[1]
        np.ndarray[np.float64_t,ndim=3] output = np.empty(target_size + (3,), dtype=np.float64)
        np.uint8_t[:,:] image_mv = image
        np.float64_t[:,:,:] output_mv = output
        np.float64_t[:,:] lut_mv = lut 
        double mn = image.min()
        int i, j, c

    with nogil:
        for i in range(output_dim0):
            for j in range(output_dim1):
                for c in range(3):
                    output_mv[i,j,c] = lut_mv[image[dim0*i/output_dim0, dim1*j/output_dim1],c]

    return output


# NEW STUFF: TODO Remove

cdef inline int int_max(int a, int b) nogil: return a if a >= b else b
cdef inline int int_min(int a, int b) nogil: return a if a <= b else b

def subsample_offset_shape(shape, size):
    return [int(shape[i]%size[i]/2 + size[i]/2)  for i in xrange(2)]

def code_parts_mmm(np.ndarray[ndim=3,dtype=np.uint8_t] X,
                   np.ndarray[ndim=3,dtype=np.uint8_t] X_unspread,
                   np.ndarray[ndim=2,dtype=np.float64_t] amps,
                   np.ndarray[ndim=4,dtype=np.float64_t] log_parts,
                   np.ndarray[ndim=4,dtype=np.float64_t] log_invparts,
                   float threshold, outer_frame=0, int collapse=1):
    """
    At each location of `X`, find the log probabilities for each part and location. Outputs these part assignments in the same data dimensions as `X`. Neighborhoods of `X` with edge counts lower than `threshold` are regarded as background and assigned zero.

    Parameters
    ----------
    X : ndarray[ndim=3,dtype=np.uint8]
        The first two dimensions of the array specify locations. The last one specifies a binary edge type. The value ``X[s,t,e]`` is 1 iff there is an edge of type `e` detected at location `(s,t)`.
    log_parts : ndarray[ndim=4]
        We have a Bernoulli mixture model defined over patches of the input image. The `log_parts` is a logarithm applied to the array of edge probability maps for each part. Array of shape `(K, S, T, E)`, where `K` is the number of mixture component, `S` and `T` the shape of the data, and `E` the number of edges. The value of ``log_parts[k,s,t,e]`` is the log probability of observing an edge `e` at location `(s,t)`, conditioned on the mixture component being `k`.
    log_invparts : ndarray[ndim=4]
        Preprocessed inverse of `log_parts`, i.e. ``log(1-exp(log_parts))``.
    threshold : int
        The least number of edges in a patch to reject the null background hypothesis.
    outer_frame : int
        Remove a frame of this thickness when checking the threshold. If the parts are 5 x 5, and this is set to 1, then only the center 3 x 3 is used to count edges when compared to the threshold. 
    
    Returns
    -------
    out_map : ndarray[ndim=3] 
        Array of shape `(S, T, K+1)`. There are two cases, either the third dimension is `(0, -inf, -inf, ...)`, when there are insufficient edges in the neighborhood of a location. Otherwise, `out_map[s,t,i+1]` is the log likelihood of part `i` at location `(s,t)`. Additionally, `out_map[s,t,0]` is equal to `-inf`.
    """

    cdef unsigned int num_parts = log_parts.shape[0]
    cdef unsigned int part_x_dim = log_parts.shape[1]
    cdef unsigned int part_y_dim = log_parts.shape[2]
    cdef unsigned int part_z_dim = log_parts.shape[3]
    cdef unsigned int X_x_dim = X.shape[0]
    cdef unsigned int X_y_dim = X.shape[1]
    cdef unsigned int X_z_dim = X.shape[2]
    cdef unsigned int new_x_dim = X_x_dim - part_x_dim + 1
    cdef unsigned int new_y_dim = X_y_dim - part_y_dim + 1
    cdef unsigned int i_start,j_start,i_end,j_end,i,j,z,k, cx0, cx1, cy0, cy1 
    cdef float count
    cdef unsigned int i_frame = <unsigned int>outer_frame
    cdef np.float64_t NINF = np.float64(-np.inf)
    # we have num_parts + 1 because we are also including some regions as being
    # thresholded due to there not being enough edges
    
    cdef np.ndarray[dtype=np.int32_t, ndim=2] out_map = -np.ones((new_x_dim,
                                                                 new_y_dim), dtype=np.int32)
    cdef np.ndarray[dtype=np.float64_t, ndim=1] vs = np.ones(num_parts, dtype=np.float64)
    cdef np.float64_t[:] vs_mv = vs

    #cdef np.ndarray[dtype=np.float64_t, ndim=1] vs_alt = np.ones(num_parts, dtype=np.float64)
    #cdef np.float64_t[:] vs_alt_mv = vs_alt

    parts = np.exp(log_parts)
    #parts_alt = np.tile(np.apply_over_axes(np.mean, parts, [1, 2]), (part_x_dim, part_y_dim, 1))
    #parts_alt = np.tile(np.apply_over_axes(np.mean, parts, [1, 2, 3]), (part_x_dim, part_y_dim, part_z_dim))

    #cdef np.ndarray[dtype=np.float64_t, ndim=4] part_logits_alt = np.rollaxis((np.log(parts_alt) - np.log(1 - parts_alt)).astype(np.float64), 0, 4).copy()
    #cdef np.ndarray[dtype=np.float64_t, ndim=1] constant_terms_alt = np.apply_over_axes(np.sum, np.log(1 - parts_alt).astype(np.float64), [1, 2, 3]).ravel()

    #cdef np.float64_t[:,:,:,:] part_logits_alt_mv = part_logits_alt
    #cdef np.float64_t[:] constant_terms_alt_mv = constant_terms_alt

    cdef np.uint8_t[:,:,:] X_mv = X
    cdef np.uint8_t[:,:,:] X_unspread_mv = X_unspread
    cdef np.float64_t[:,:] amps_mv = amps

    cdef np.ndarray[dtype=np.float64_t, ndim=4] part_logits = np.rollaxis((log_parts - log_invparts).astype(np.float64), 0, 4).copy()

    cdef np.ndarray[dtype=np.float64_t, ndim=1] constant_terms = np.apply_over_axes(np.sum, log_invparts.astype(np.float64), [1, 2, 3]).ravel()
    cdef np.float64_t[:,:,:,:] part_logits_mv = part_logits
    cdef np.float64_t[:] constant_terms_mv = constant_terms

    cdef np.int32_t[:,:] out_map_mv = out_map

    cdef np.ndarray[dtype=np.float64_t, ndim=2] _integral_counts = np.zeros((X_x_dim+1, X_y_dim+1), dtype=np.float64)
    cdef np.float64_t[:,:] integral_counts = _integral_counts

    cdef np.float64_t max_value, v
    cdef int max_index 

    # The first cell along the num_parts+1 axis contains a value that is either 0
    # if the area is deemed to have too few edges or min_val if there are sufficiently many
    # edges, min_val is just meant to be less than the value of the other cells
    # so when we pick the most likely part it won't be chosen

    # Build integral image of edge counts
    # First, fill it with edge counts and accmulate across
    # one axis.
    for i in range(X_x_dim):
        for j in range(X_y_dim):
            count = amps_mv[i,j] / <float>(X_x_dim * X_y_dim)
            integral_counts[1+i,1+j] = integral_counts[1+i,j] + count
    # Now accumulate the other axis
    for j in range(X_y_dim):
        for i in range(X_x_dim):
            integral_counts[1+i,1+j] += integral_counts[i,1+j]

    # Code parts
    for i_start in range(X_x_dim-part_x_dim+1):
        i_end = i_start + part_x_dim
        for j_start in range(X_y_dim-part_y_dim+1):
            j_end = j_start + part_y_dim

            # Note, integral_counts is 1-based, to allow for a zero row/column at the zero:th index.
            cx0 = i_start+i_frame
            cx1 = i_end-i_frame
            cy0 = j_start+i_frame
            cy1 = j_end-i_frame
            count = integral_counts[cx1, cy1] - \
                    integral_counts[cx0, cy1] - \
                    integral_counts[cx1, cy0] + \
                    integral_counts[cx0, cy0]

            if threshold <= count:
                vs[:] = constant_terms
                #vs_alt[:] = constant_terms_alt
                with nogil:
                    for i in range(part_x_dim):
                        for j in range(part_y_dim):
                            for z in range(X_z_dim):
                                if X_mv[i_start+i,j_start+j,z]:
                                    for k in range(num_parts):
                                        vs_mv[k] += part_logits_mv[i,j,z,k]
                                        #vs_alt_mv[k] += part_logits_alt_mv[i,j,z,k]

                max_index = vs.argmax()
                #if vs_mv[max_index] - vs_alt_mv[max_index] >= accept_threshold:
                out_map_mv[i_start,j_start] = max_index / collapse
                
                #out_map_mv[i_start,j_start] = vs.argmax() / collapse

    return out_map

def extract_parts(edges, unspread_edges, amps, log_parts, log_invparts, float threshold, outer_frame=0, spread_radii=(4, 4), subsample_size=(4, 4), int collapse=1):
    cdef:
        int num_feats = log_parts.shape[0]//collapse
        np.ndarray[np.int32_t,ndim=2] parts = code_parts_mmm(edges, unspread_edges, amps, log_parts, log_invparts, threshold, outer_frame=outer_frame, collapse=collapse)
        np.ndarray[np.uint8_t,ndim=3] feats = np.zeros((parts.shape[0]//subsample_size[0],
                                                        parts.shape[1]//subsample_size[1],
                                                        num_feats),
                                                        dtype=np.uint8)

    offset = subsample_offset_shape((parts.shape[0], parts.shape[1]), subsample_size)

    cdef:
        np.int32_t[:,:] parts_mv = parts
        np.uint8_t[:,:,:] feats_mv = feats
  
        int spread_radii0 = spread_radii[0]
        int spread_radii1 = spread_radii[1]
        int subsample_size0 = subsample_size[0]
        int subsample_size1 = subsample_size[1]
        int feats_dim0 = feats.shape[0]
        int feats_dim1 = feats.shape[1]
        int parts_dim0 = parts.shape[0]
        int parts_dim1 = parts.shape[1]
        int offset0 = offset[0]
        int offset1 = offset[1]
        int p, x, y, i, j, i0, j0

    with nogil:
        for i in range(feats_dim0):
            for j in range(feats_dim1):
                x = offset0 + i*subsample_size0 
                y = offset1 + j*subsample_size1
                for i0 in range(int_max(x - spread_radii0, 0), int_min(x + spread_radii0+1, parts_dim0)):
                    for j0 in range(int_max(y - spread_radii1, 0), int_min(y + spread_radii1+1, parts_dim1)):
                        p = parts_mv[i0,j0]
                        if p != -1:
                            feats_mv[i,j,p] = 1

    return feats 

def find_zeros_when_mcmc_training(np.ndarray[ndim=3,dtype=np.float64_t] Xbar,
                                  np.ndarray[ndim=2,dtype=np.float64_t] LZ_counts,
                                  np.ndarray[ndim=2,dtype=np.float64_t] LZ_values,
                                  lower=-10,
                                  upper=10):


    cdef:
        int BINS = LZ_counts.shape[1] 
        int total_counts = LZ_counts[0].sum()

        np.float64_t[:,:,:] Xbar_mv = Xbar
        #np.float64_t[:,:] logit_Zs_mv = logit_Zs
        np.float64_t v, lo, up, mi, lower_real, upper_real

        np.float64_t[:,:] LZ_counts_mv = LZ_counts
        np.float64_t[:,:] LZ_values_mv = LZ_values 
        

        int dim0 = Xbar.shape[0]
        int dim1 = Xbar.shape[1]
        #int num_Z = logit_Zs.shape[0]
        int F = Xbar.shape[2]
        int f, l0, l1, i, k

        np.ndarray[ndim=3,dtype=np.float64_t] w = np.zeros((dim0, dim1, F), dtype=np.float64)
        np.float64_t[:,:,:] w_mv = w

    lower_real = <real>lower
    upper_real = <real>upper
        
    with nogil:
        for f in xrange(F): 
            for l0 in xrange(dim0):
                for l1 in xrange(dim1):
                    lo = lower_real
                    up = upper_real 

                    for i in xrange(10): 
                        mi = (lo + up) / 2.0 
                        v = 0.0
                        for k in xrange(BINS):
                            v += LZ_counts_mv[f,k] * sigmoid(mi + LZ_values_mv[f,k])
                        v /= total_counts
                        v -= Xbar[l0,l1,f]
                        if v > 0:
                            up = mi 
                        else:
                            lo = mi

                    w_mv[l0,l1,f] = mi 
    return w

cdef int _area(int bb0, int bb1, int bb2, int bb3):
    return int_max(0, (bb2 - bb0)) * int_max(0, (bb3 - bb1))

def best_bounding_box(np.ndarray[ndim=2,dtype=np.int64_t] contendors, np.ndarray[ndim=2,dtype=np.int64_t] bbs):
    cdef:
        int N = contendors.shape[0]
        int M = bbs.shape[0]
        int i, j, inflate

        np.ndarray[ndim=1,dtype=np.float64_t] scores = np.zeros(N)
        np.float64_t[:] scores_mv = scores

        np.int64_t[:,:] contendors_mv = contendors
        np.int64_t[:,:] bbs_mv = bbs 

        int bb0, bb1, bb2, bb3, area1, area2, area_union, area_intersection, ok
        float metric, score

    for i in xrange(N):
        # Calculate score
        score = 0.0

        for j in xrange(M):
            bb0 = int_max(contendors_mv[i,0], bbs_mv[j,0])
            bb1 = int_max(contendors_mv[i,1], bbs_mv[j,1])
            bb2 = int_min(contendors_mv[i,2], bbs_mv[j,2])
            bb3 = int_min(contendors_mv[i,3], bbs_mv[j,3])

            area1 = _area(contendors_mv[i,0], contendors_mv[i,1], contendors_mv[i,2], contendors_mv[i,3])
            area2 = _area(bbs_mv[j,0], bbs_mv[j,1], bbs_mv[j,2], bbs_mv[j,3])
            area_intersection = _area(bb0, bb1, bb2, bb3)
            area_union = area1 + area2 - area_intersection

            metric = <float>area_intersection / (<float>area_union + 0.001)
            score += -<float>(metric > 0.5) - 0.01 * metric

        scores_mv[i] = score
    return np.argmin(scores)
