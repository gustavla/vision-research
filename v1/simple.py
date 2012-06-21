import numpy as np
import mnist

def train(path='.'):
    params = None 
    for i in range(10):
        images, _ = mnist.read([i], 'training', path)
        size = images[0].shape
        if params is None:
            params = np.zeros((10, size[0], size[1]))
        params[i] = images.mean(axis=0) 
    return params
    
# Requires numpy 1.7
#@np.vectorize(classify, excluded=['params'])
def classify(image, params):
    sqerrors = np.zeros((10))
    for i in range(10):
        sqerrors[i] = np.sum(np.fabs(params[i] - image)**2.0)
    return np.argmin(sqerrors)


def evaluate(path='.'):
    params = train(path)
    images, labels = mnist.read(range(10), 'testing', path)
    c = 0
    N = images.shape[0]
    for i in range(N):
        c += int(classify(images[i], params) == labels[i])
    return c/float(N)
