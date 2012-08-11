
import amitgroup as ag
import numpy as np
import sys
import time

try:
    features_filename = sys.argv[1]
    mixtures_filename = sys.argv[2]
    output_filename = sys.argv[3]

except IndexError:
    print "<features filename> <mixture filename> <output filename>"
    sys.exit(0)

PLOT = False

features_file = np.load(features_filename)
mixtures_file = np.load(mixtures_filename)
all_templates = mixtures_file['templates']
all_affinities = mixtures_file['affinities']

if PLOT:
    plw = ag.plot.PlottingWindow(figsize=(5, 9), subplots=(8, 3))

M = all_affinities.shape[1]

sh = (10, M) + ag.util.DisplacementFieldWavelet.shape_for_size(all_templates.shape[2:4])
means = np.empty(sh)
variances = np.empty(sh)

llh_sh = (10, M)
llh_means = np.empty(llh_sh)
llh_variances = np.empty(llh_sh)

#means = [[] for i in range(M)]
#variances = [[] for i in range(M)]
for d in range(10):
    entries = [[] for i in range(M)]
    slices = [[] for i in range(M)]
    all_features = features_file[str(d)] 
    N = len(all_features)
    us = []
    for i in range(N):
        if PLOT and not plw.tick():
            sys.exit(0) 
        affinities = all_affinities[i]
        m = np.argmax(affinities)
        F = np.rollaxis(all_templates[d,m], axis=2)
        I = np.rollaxis(all_features[i], axis=2).astype(float)

        t1 = time.time()
        imdef, info = ag.stats.bernoulli_deformation(F, I, penalty=100.0, gtol=0.1, maxiter=1000, last_level=3, wavelet='db8', debug_plot=False)
        t2 = time.time()
        print i, "time:", (t2-t1)
        #imdef, info = ag.ml.bernoulli_deformation(F, I, penalty=100.0, stepsize_scale_factor=0.1, tol=0.000001, maxiter=1000, last_level=4, wavelet='db2', debug_plot=False)

        #us.append(imdef.u)
        entries[m].append(imdef.u)
        Fdef = np.asarray([
            imdef.deform(F[j]) for j in range(8)
        ])
        slices[m].append(Fdef - I)
        
        if PLOT:
            for j in range(8):
                plw.imshow(I[j], subplot=j*3)
                plw.imshow(F[j], subplot=j*3+1)
                plw.imshow(imdef.deform(F[j]), subplot=j*3+2)

    for m in range(M):
        data = np.asarray(entries[m])
        assert len(data) > 0, "Need more data!" 

        print means.shape, data.shape
        # Prior
        means[d, m] = data.mean(axis=0) 
        variances[d, m] = data.var(axis=0) 

        # Likelihood
        values = np.asarray(slices[m]).flatten()
        np.save('tmp-values.{0}.npy'.format(m), values)

        llh_means[d, m] = values.mean()
        llh_variances[d, m] = values.var()
         
    #us = np.asarray(us)
    #means.append(us.mean())
    #variances.append(us.var())

np.savez(output_filename, prior_mean=means, prior_var=variances, llh_mean=llh_means, llh_var=llh_variances)
