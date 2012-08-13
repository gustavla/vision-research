
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
if args.inspect is not None:
    inspect = args.inspect[0]
else:
    inspect = None
digit = args.digit

features_file = np.load(features_filename)
mixtures_file = np.load(mixtures_filename)
all_templates = mixtures_file['templates']
all_affinities = mixtures_file['affinities']

M = all_affinities.shape[1]

if digit is not None:
    digits = [digit]
    shape = (1, M)
    d0 = digit
else:
    digits = range(10)
    shape = (10, M)
    d0 = 0

sh = shape + ag.util.DisplacementFieldWavelet.shape_for_size(all_templates.shape[2:4])
means = np.empty(sh)
variances = np.empty(sh)

llh_sh = shape 
llh_means = np.empty(llh_sh)
llh_variances = np.empty(llh_sh)

for d in digits:
    entries = [[] for i in range(M)]
    slices = [[] for i in range(M)]
    all_features = features_file[str(d)] 
    n0 = 0
    n1 = len(all_features)
    if inspect is not None:
        n0 = inspect
        n1 = n0+1 

    n1 = 30 

    us = []
    for i in range(n0, n1):
        affinities = all_affinities[i]
        m = np.argmax(affinities)
        F = np.rollaxis(all_templates[d,m], axis=2)
        I = np.rollaxis(all_features[i], axis=2).astype(float)

        x, y = ag.util.DisplacementFieldWavelet.meshgrid_for_shape(F.shape[1:])

        t1 = time.time()
        imdef, info = ag.stats.bernoulli_deformation(F, I, penalty=10000.0, rho=2.0, gtol=0.1, maxiter=10, start_level=1, last_level=1, wavelet='db4', debug_plot=PLOT)
        t2 = time.time()
        print "{0}.{1} (time = {2}".format(d, i, t2-t1)

        if imdef is None:
            sys.exit(0) 

        entries[m].append(imdef.u)
        Fdef = np.asarray([
            imdef.deform(F[j]) for j in range(8)
        ])
        slices[m].append(Fdef - I)

    for m in range(M):
        data = np.asarray(entries[m])
        assert len(data) > 0, "Need more data!" 

        print means.shape, data.shape
        # Prior
        print d, m, means.shape
        means[d-d0, m] = data.mean(axis=0) 
        variances[d-d0, m] = data.var(axis=0) 

        # Likelihood
        values = np.asarray(slices[m]).flatten()
        np.save("tmp-values.{0}.npy".format(m), values)

        llh_means[d-d0, m] = values.mean()
        llh_variances[d-d0, m] = values.var()
         
np.savez(output_filename, prior_mean=means, prior_var=variances, llh_mean=llh_means, llh_var=llh_variances)
