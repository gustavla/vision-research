import numpy as np
import amitgroup as ag
import sys
import os

try:
    dataset = sys.argv[1] 
    filename = sys.argv[2]
    k = int(sys.argv[3])
    inflate = 'inflate' in sys.argv 

except IndexError:
    print "(training|testing) <output filename> <k> [inflate]"
    sys.exit(0)

digit_features = {} 
for d in range(10):
    print(d)
    digits, _ = ag.io.load_mnist(dataset, [d])
    features = ag.features.bedges(digits, k=k, inflate=inflate)
    digit_features[str(d)] = features


np.savez(filename, **digit_features)

