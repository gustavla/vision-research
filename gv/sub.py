from __future__ import division
import numpy as np

def subsample_size(data, size):
    return tuple([data.shape[i]//size[i] for i in xrange(2)])

def subsample_size_new(shape, size):
    return tuple([shape[i]//size[i] for i in xrange(2)])

def iround(x):
    return int(round(x))

def subsample_offset_shape(shape, size):
    return [int(shape[i]%size[i]/2 + size[i]/2)  for i in xrange(2)]

def subsample_offset(data, size):
    return [int(data.shape[i]%size[i]/2 + size[i]/2)  for i in xrange(2)]

def subsample_offset2(data, size):
    return [size[i]//2  for i in xrange(2)]

def subsample(data, size, skip_first_axis=False):
    # TODO: Make nicer
    if skip_first_axis:
        offsets = subsample_offset(data[0], size) 
    else:
        offsets = subsample_offset(data, size) 

    if skip_first_axis:
        return data[:,offsets[0]::size[0],offsets[1]::size[1]]
    else:
        return data[offsets[0]::size[0],offsets[1]::size[1]]

def erase(x, y, psize):
    offset = np.asarray(subsample_offset(x, psize)) - np.asarray(subsample_offset2(x, psize))
        
    for i in xrange(y.shape[0]):
        for j in xrange(y.shape[1]):
            if y[i,j] == 0:
                edges = [[0, 0], [0, 0]] 
                for k in xrange(2):
                    it = (i,j)[k]
                    edges[k][0] = offset[k]+it*psize[k]
                    edges[k][1] = offset[k]+(it+1)*psize[k]

                    if it == 0:
                        edges[k][0] = None 
                    elif it == y.shape[k]-1:
                        edges[k][1] = None

                x[edges[0][0]:edges[0][1], 
                  edges[1][0]:edges[1][1]] = 0 

            #x[offset[0]+i*psize[0]:offset[0]+(i+1)*psize[0], 
            #  offset[1]+j*psize[1]:offset[1]+(j+1)*psize[1]] &= y[i,j] 
    return x

