
def plotb(coefs_files, lmb0, b0, digit=0, mixture=0, axis=0):
    import numpy as np
    import matplotlib.pylab as plt
    import amitgroup as ag

    for coefs_file in coefs_files:
        coefs_data = np.load(coefs_file)
        var = coefs_data['prior_var']
        samples = coefs_data['samples']

        var_flat = ag.util.wavelet.smart_flatten(var[digit,mixture,axis])
        last_i = len(var_flat)-1
        plt.xlim((0, last_i))

        #imdef = ag.util.DisplacementFieldWavelet((32, 32), 'db4', penalty=100
        if lmb0 > 0:
            plt.semilogy([0, last_i], [lmb0]*2, label="Prior {0}".format(coefs_file.name))

        if b0 > 0 and len(coefs_files):
            if b0 and lmb0 and samples is not None:
                new_var = (b0 + samples*var/2) / (b0 * lmb0 + samples/2)
            new_flat = ag.util.wavelet.smart_flatten(new_var[digit,mixture,axis])
            plt.semilogy(1/new_flat, label="Posterior {0}".format(coefs_file.name))

        plt.semilogy(1/var_flat, label="ML {0}".format(coefs_file.name))
    plt.legend(loc=0)
    plt.show()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot posterior variance')
    parser.add_argument('coefficients', nargs='+', metavar='<coef file>', type=argparse.FileType('rb'), help='Filename of model coefficients')
    parser.add_argument('-l', dest='lmb', metavar='LAMBDA', nargs=1, default=[0.0], type=float, help='Prior lambda peak of Gamma distribution')
    parser.add_argument('-b', metavar='B', nargs=1, default=[0.0], type=float, help='Prior hypercoefficient b of Gamma distribution')
    parser.add_argument('-i', '--index', nargs=3, metavar=('DIGIT', 'MIXTURE', 'AXIS'), default=(0, 0, 0), type=int, help='Index of data, with choice of DIGIT, MIXTURE and AXIS. Defaults to (0, 0, 0).')

    args = parser.parse_args()
    coef_files = args.coefficients
    lmb0 = args.lmb[0]
    b0 = args.b[0]
    digit, mixture, axis = args.index

    plotb(coef_files, lmb0, b0, digit, mixture, axis)
