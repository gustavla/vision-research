from __future__ import print_function
import sys
import os
import glob
import numpy as np
import matplotlib.pylab as plt

try:
    path_input = sys.argv[1]
    path_output = sys.argv[2]
except IndexError:
    raise ValueError("Must specify path (with wildcards) for training data and output") 


try:
    files = glob.glob(os.path.expanduser(path_input))
except IOError:
    print("Could not fetch files from {0}".format(path_input))
    sys.exit(0)

res = None

for i, f in enumerate(files):
    print(i, f)
    data = plt.imread(f)
    if res is None:
        res = np.zeros((len(files),) + data.shape[:2]) 
        print(len(f))
    print(data.shape)
    res[i] = data[:,:,:3].mean(axis=2) 

try:
    np.savez(path_output, data=res)
except IOError:
    print("Could not save to {0}".format(output))    

print("SAVED (to {0})".format(path_output))
