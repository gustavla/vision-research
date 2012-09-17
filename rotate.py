import numpy as np
import itertools
#x = np.arange(9).reshape((3, 3))
#x = np.arange(25).reshape((5, 5))
x = np.array([
    [0,0,0,0,1],
    [0,1,1,1,0],
    [0,1,1,1,0],
    [0,1,1,1,0],
    [1,0,0,0,0],
])

#def _rotate(x, j):
    #for i in xrange(j):
        #x = _rotate_once(x)
    
translate = {
    (1,1): (1,1),
    (0,0): (1,0),
    (1,0): (2,0),
    (2,0): (2,1),
    (2,1): (2,2),
    (2,2): (1,2),
    (1,2): (0,2),
    (0,2): (0,1),
    (0,1): (0,0),
}

def _rotate(x, steps):
    #y = np.empty_like(x)
    y = np.ones_like(x) * -1
    c = x.shape[0]//2
    s = x.shape[0]
    for i, j in itertools.product(*[range(s) for s in x.shape]):
        step = max(abs(i-c), abs(j-c))
        #y[translate[i,j]] = x[i,j]
        ni = i
        nj = j
        if i == c+step:
            nj += step
            if nj > c+step:
                ni += (c+step-nj) 
                nj = c+step
        elif i == c-step:
            nj -= step
            if nj < c-step:
                ni += (c-step-nj) 
                nj = c-step 
        elif j == c-step:
            ni += step
            if ni >= c+step:
                nj -= (c+step-ni)
                ni = c+step 
        elif j == c+step:
            ni -= step
            if ni < c-step:
                nj -= (c-step-ni) 
                ni = c-step
        print (ni, nj), (i, j)
    
        y[ni,nj] = x[i,j]
                
    return y

print x
print _rotate(x, 1)
