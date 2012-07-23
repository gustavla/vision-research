
import numpy as np

_mul = False

class _matrix_multiplication(object):
    def __enter__(self):
        global _mul
        _mul = True
    
    def __exit__(self, *args):
        print args
        global _mul
        _mul = False  

matrix_multiplication = _matrix_multiplication()

class new_array(np.ndarray):
    def __init__(self, x):
        self = np.array(x) 

    def __mul__(self, o):
        if _mul:
            return np.array.__mul__(self, o)
        else:
            return np.dot(self, o) 

x = np.eye(2, 2)
y = np.ones((2, 2)) * 2

#x = new_array([[1,0],[0,1]])
#y = new_array(y)

z1 = x * y

with matrix_multiplication as m:
    z2 = x * y

print z1
print z2
