import numpy as np
import amitgroup as ag
import sys
import os

try:
    dataset = sys.argv[1] 
except IndexError:
    dataset = 'training'

digit_features = {} 
for d in range(10):
    print(d)
    digits, _ = ag.io.load_mnist(dataset, [d])
    features = ag.features.bedges(digits, k=5, inflate=True)
    digit_features[str(d)] = features

path = "/var/tmp/local"

np.savez(os.path.join(path, 'mnist-{0}-features-4'.format(dataset)), **digit_features)

