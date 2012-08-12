
import amitgroup as ag
import numpy as np
import sys
import time
import argparse

parser = argparse.ArgumentParser(description='Train coefficients')
parser.add_argument('features', metavar='<features file>', type=str, help='Filename of features')
parser.add_argument('mixtures', metavar='<mixtures file>', type=str, help='Filename of mixtures')
parser.add_argument('output', metavar='<output file>', type=str, help='Filename of output')
parser.add_argument('-p', '--plot', action='store_true', help='Plot using pygame')
parser.add_argument('-i', '--inspect', nargs=1, type=int, help='Inspect a single element')
parser.add_argument('-d', '--digit', nargs='?', type=int, help='Process only one digit')

args = parser.parse_args()
features_filename = args.features
mixtures_filename = args.mixtures
output_filename = args.output
PLOT = args.plot
inspect = args.inspect[0]
digit = args.digit

features_file = np.load(features_filename)
mixtures_file = np.load(mixtures_filename)
all_templates = mixtures_file['templates']
all_affinities = mixtures_file['affinities']

#if PLOT:
#    plw = ag.plot.PlottingWindow(figsize=(5, 9), subplots=(8, 3))

M = all_affinities.shape[1]

sh = (10, M) + ag.util.DisplacementFieldWavelet.shape_for_size(all_templates.shape[2:4])
means = np.empty(sh)
variances = np.empty(sh)

llh_sh = (10, M)
llh_means = np.empty(llh_sh)
llh_variances = np.empty(llh_sh)

if digit is not None:
    digits = [digit]
else:
    digits = range(10)

#means = [[] for i in range(M)]
#variances = [[] for i in range(M)]
for d in digits:
    entries = [[] for i in range(M)]
    slices = [[] for i in range(M)]
    all_features = features_file[str(d)] 
    n0 = 0
    n1 = len(all_features)
    if inspect is not None:
        n0 = inspect
        n1 = n0+1 

    us = []
    for i in range(n0, n1):
        #if PLOT and not plw.tick():
        #    sys.exit(0) 
        affinities = all_affinities[i]
        m = np.argmax(affinities)
        F = np.rollaxis(all_templates[d,m], axis=2)
        I = np.rollaxis(all_features[i], axis=2).astype(float)

        t1 = time.time()
        imdef, info = ag.stats.bernoulli_deformation(F, I, penalty=100.0, gtol=0.1, maxiter=10, last_level=3, wavelet='db8', debug_plot=PLOT)
        t2 = time.time()
        print i, "time:", (t2-t1)
        #imdef, info = ag.ml.bernoulli_deformation(F, I, penalty=100.0, stepsize_scale_factor=0.1, tol=0.000001, maxiter=1000, last_level=4, wavelet='db2', debug_plot=False)

        #us.append(imdef.u)
        entries[m].append(imdef.u)
        Fdef = np.asarray([
            imdef.deform(F[j]) for j in range(8)
        ])
        slices[m].append(Fdef - I)
        
        #if PLOT:
        #    for j in range(8):
        #        plw.imshow(I[j], subplot=j*3)
        #        plw.imshow(F[j], subplot=j*3+1)
        #        plw.imshow(imdef.deform(F[j]), subplot=j*3+2)

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
