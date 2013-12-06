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

    with nogil:
        for i in range(steps_x):
            for j in range(steps_y):
                v = 0
                for sx in range(kernel_d0):
                    for sy in range(kernel_d1):
                        for f in range(num_feat):
                            v += data[i+sx,j+sy,f] * kernel[sx,sy,f]
                response[i,j] = v

    return response_

def multifeature_correlate2d_with_indices(np.ndarray[mybool,ndim=3] data_, np.ndarray[real,ndim=3] kernel_, np.ndarray[np.int32_t, ndim=2] indices_):
    assert data_.shape[0] >= kernel_.shape[0]
    assert data_.shape[1] >= kernel_.shape[1]
    assert indices_.shape[1] == 3
    cdef:
        int data_d0 = data_.shape[0]
        int data_d1 = data_.shape[1]
        int kernel_d0 = kernel_.shape[0]
        int kernel_d1 = kernel_.shape[1]
        int steps_x = (data_d0 - kernel_d0) + 1
        int steps_y = (data_d1 - kernel_d1) + 1
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
        int i, j, ii, sx, sy, f

    #with nogil:
    #    for i in range(steps_x):
    #        for j in range(steps_y):
    #            v = 0
    #            for ii in range(num_indices):
    #                sx = indices[ii,0]     
    #                sy = indices[ii,1]
    #                f = indices[ii,2]
    #                v += data[i+sx,j+sy,f] * kernel[sx,sy,f]
    #                #for f in range(num_feat):
    #                    #v += data[i+sx,j+sy,f] * kernel[sx,sy,f]
    #            response[i,j] = v

    with nogil:
        for ii in range(num_indices):
            sx = indices[ii,0]     
            sy = indices[ii,1]
            f = indices[ii,2]
            for i in range(steps_x):
                for j in range(steps_y):
                    v = 0
                    v += data[i+sx,j+sy,f] * kernel[sx,sy,f]
                    #for f in range(num_feat):
                        #v += data[i+sx,j+sy,f] * kernel[sx,sy,f]
                    response[i,j] += v

    return response_

def multifeature_correlate2d_multi(np.ndarray[mybool,ndim=3] data_, np.ndarray[real,ndim=4] kernels_, np.ndarray[np.int32_t,ndim=2] bkgcomps_):
    assert data_.shape[0] > kernels_.shape[1]
    assert data_.shape[1] > kernels_.shape[2]
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
