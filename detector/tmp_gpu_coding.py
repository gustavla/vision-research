from __future__ import division
import argparse
import gv

parser = argparse.ArgumentParser(description='Test response of model')
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file')

args = parser.parse_args()
model_file = args.model

import numpy as np
import os
import glob
import amitgroup as ag
import pyopencl as cl

detector = gv.Detector.load(model_file)

S = 200
density = 0.1

rs = np.random.RandomState(0)
edges = (rs.uniform(size=(200, 200, 4)) < density).astype(np.uint8)
#im = next(glob.iglob(os.path.expandvars('$VOC_DIR/JPEGImages/*.jpg')))

eps = 0.025
parts = np.clip(detector.descriptor.parts, eps, 1 - eps)
log_probs = np.log(parts)
log_invprobs = np.log(1 - parts)

spread_edges = edges

#print help(ag.features.code_parts)
coded = ag.features.code_parts(spread_edges, edges, log_probs, log_invprobs, 0)

# Now, do GPU code parts
KERNEL_CODE = """
// Thread block size
#define BLOCK_SIZE %(block_size)d

// Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
#define WA %(w_a)d // Matrix A width
#define HA %(h_a)d // Matrix A height
#define WB %(w_b)d // Matrix B width
#define HB WA  // Matrix B height
#define WC WB  // Matrix C width
#define HC HA  // Matrix C height
         
__kernel __attribute__((reqd_work_group_size(BLOCK_SIZE,BLOCK_SIZE,1))) 
void
matrixMul( __global int* X, 
           __global float* log_parts, 
           __global float* log_invparts, 
           __global float* output) {

}

print coded.shape

print detector.descriptor.parts.shape
