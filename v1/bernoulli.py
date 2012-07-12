
import amitgroup as ag
import amitgroup.features
import numpy as np
import os
import matplotlib.pylab as plt

def classify(image, F):
    

def zeropad(array, pad_width):
    shape = list(array.shape)
    slices = []
    shape = [array.shape[i] + pad_width[i] * 2 for i in range(len(array.shape))]
    slices = [slice(pad_width[i], pad_width[i]+array.shape[i]) for i in range(len(array.shape))]
    new_array = np.zeros(shape)
    new_array[slices] = array
    return new_array

def main():
    # Load features
    try:
        F = np.load('F.npz')['F']
    except IOError:
        digits = [ag.io.load_mnist('training', os.environ['MNIST'], [i])[0] for i in range(10)]
        feats = [ag.features.bedges(zeropad(d, (0, 2, 2))) for d in digits] 
        F = np.array([[feats[i][:,:,:,j].mean(axis=0) for j in range(8)] for i in range(10)])
        np.savez('F.npz', F=F)

    #plt.imshow(F[0,0], interpolation='nearest', cmap=plt.cm.gray_r)
    #plt.show()

    testdata, labels = ag.io.load_mnist('testing', os.environ['MNIST']) 

    for test_i in range(3):
        label = classify(testdata[test_i], F)
        print "RESULT: {0} {2} {1}".format(label, labels[test_i], ['failed', 'correct'][label == labels[test_i]])

if __name__ == '__main__':
    main()
