from __future__ import division

def plot(coefs_files, digit=0, mixture=0, axis=0):
    import numpy as np
    import matplotlib.pylab as plt
    import amitgroup as ag

    for coefs_file in coefs_files:
        coefs_data = np.load(coefs_file)
        var = coefs_data['prior_var']
        samples = coefs_data['samples']
        llh_var = coefs_data['llh_var']

        var_flat = ag.util.wavelet.smart_flatten(var[digit,mixture,axis])
        last_i = len(var_flat)-1
        plt.xlim((0, last_i))

        #imdef = ag.util.DisplacementFieldWavelet((32, 32), 'db4', penalty=100

        if len(coefs_files) == 1:
            add = ""
        else:   
            add = " ({0})".format(coefs_file.name)

        plt.subplot(121)
        plt.semilogy(1/var_flat, label="ML"+add)
        plt.legend(loc=0)
        plt.xlabel('Coefficient')
        plt.ylabel('Precision $\lambda$')

        plt.subplot(122)
        plt.imshow(1/llh_var[digit,mixture])
        plt.xlabel('likelihood precision')
        plt.colorbar()
    plt.show()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot coefficients for intensity-based data model')
    parser.add_argument('coefficients', nargs='+', metavar='<coef file>', type=argparse.FileType('rb'), help='Filename of model coefficients')
    parser.add_argument('-i', '--index', nargs=3, metavar=('DIGIT', 'MIXTURE', 'AXIS'), default=(0, 0, 0), type=int, help='Index of data, with choice of DIGIT, MIXTURE and AXIS. Defaults to (0, 0, 0).')

    args = parser.parse_args()
    coef_files = args.coefficients
    digit, mixture, axis = args.index

    plot(coef_files, digit, mixture, axis)
