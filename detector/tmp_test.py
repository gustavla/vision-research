
import numpy as np

sh = (100, 30, 40, 50)

X = np.random.normal(size=sh[0])
Y = np.random.normal(size=sh)

import gv.fast

res = gv.fast.correlate_abunch(X, Y)
print res[0,0,0]
print np.corrcoef(X, Y[:,0,0,0])
print np.isnan(res).sum()
