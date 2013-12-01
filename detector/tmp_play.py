

sh = (2, 3, 4)
sh2 = (4, 2)
X = np.arange(np.prod(sh)).reshape(sh)
Y = np.ones(np.prod(sh2)).reshape(sh2)

X[i,j,k] = Y[k,i]

i, j, k = A.ii[:2]
A(X)[i, j, k] = A(Y)[k, i]
