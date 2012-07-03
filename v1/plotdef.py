
import amitgroup as ag
import amitgroup.ml.imagedefw
import numpy as np
import matplotlib.pylab as plt
import pywt

images = np.load("data/nines.npz")['images']
shifted = np.zeros(images[0].shape)    
shifted[:-3,:] = images[0,3:,:]

im1, im2 = np.zeros((32, 32)), np.zeros((32, 32))

#im1, im2 = images[0], shifted
im1[:28,:28] = images[0]

face = ag.io.load_example('faces')[0]
face = face[::-1,:]

if 0:
    face = np.zeros((32,32))

    for i in range(32):
        face[8,i] = 0.5
        face[24,i] = 0.5
        face[i,8] = 0.5
        face[i,24] = 0.5
        face[16,i] = 1.0
        face[i,16] = 1.0
        face[i,1] = 0.7
        face[i,3] = 0.4
        face[i,5] = 0.4
        face[i,30] = 0.7

levels = 5
scriptNs = map(len, pywt.wavedec(range(32), 'db2', level=levels)) + [0]

#u = np.zeros((2, d, d))
ue = []
for q in range(2):
    u0 = [np.zeros((scriptNs[0],)*2)]
    for a in range(1, levels):
        sh = (scriptNs[a],)*2
        u0.append((np.zeros(sh), np.zeros(sh), np.zeros(sh)))
    ue.append(u0)

#u[0][0][2,0] = -5.5

#u[0]
import pickle
u2 = pickle.load(open('u.p', 'rw')) # np.load('u')['u']

from copy import deepcopy
u = deepcopy(u2)
for q in range(2):
    #u[q][0][:] = 0.0 
    for a in range(1, len(u[q])):
        for alpha in range(3):
            u[q][a][alpha][:,:] = 0.0 

#u[0][0][:] = 0.0
#print u[1][0]
#u[1][0][0,0] = 0.0



levcoefs = np.zeros(len(u[0]))
for q in range(2):
    for i in range(len(u[q])):
        if i == 0:
            levcoefs[i] += (u[q][i]**2 ).sum()
        else:
            for alpha in range(3):
                levcoefs[i] += (u[q][i][alpha]**2 ).sum()

#print levcoefs

from itertools import product
shape = im1.shape
xs = np.empty(shape + (2,))
for x0, x1 in product(range(shape[0]), range(shape[1])): 
    xs[x0,x1] = np.array([float(x0)/(shape[0]), float(x1)/shape[1]])
defmap = ag.ml.imagedefw.deform_map(xs, u)
face2 = ag.ml.imagedefw.deform(face, u)  

d = dict(origin='lower', interpolation='nearest', cmap=plt.cm.gray)
plt.figure(figsize=(9,9))
plt.subplot(221)
plt.imshow(face, **d)
plt.subplot(222)
plt.imshow(face2, **d)
plt.subplot(223)
plt.plot(levcoefs)
#plt.pcolor(u[1][0])
#plt.colorbar()
plt.subplot(224)
plt.quiver(xs[:,:,1], xs[:,:,0], defmap[:,:,1], defmap[:,:,0])
plt.show()
