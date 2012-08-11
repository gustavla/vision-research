import numpy as np
import amitgroup as ag
import sys
import os

try:
    dataset = sys.argv[1] 
    filename = sys.argv[2]
    k = int(sys.argv[3])
    N0 = int(sys.argv[4])
    N1 = int(sys.argv[5])
    inflate = 'inflate' in sys.argv 

except IndexError:
    print "(training|testing) <output filename> <k> <N-start> <N-stop> [inflate]"
    sys.exit(0)

digit_features = {} 
for d in range(10):
    print(d)
    digits, _ = ag.io.load_mnist(dataset, [d])
    digits = digits[N0:N1]

    digits_padded = np.zeros((len(digits),) + (32, 32))
    digits_padded[:,2:-2,2:-2] = digits

    features = ag.features.bedges(digits_padded, k=k, inflate=inflate)
    digit_features[str(d)] = features

np.savez(filename, **digit_features)

