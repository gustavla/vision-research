
def plotb(coefs_files, eta, rho, b0, digit=0, mixture=0, axis=0):
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

        if len(coefs_files) == 1:
            add = ""
        else:   
            add = " ({0})".format(coefs_file.name)

        if eta > 0:
            lmb0 = ag.util.DisplacementFieldWavelet.make_lambdas((32, 32), 3, eta=eta, rho=rho)
            flat_lmb = ag.util.wavelet.smart_flatten(lmb0)
            plt.semilogy(flat_lmb, label="$\lambda_0$"+add)

            if b0 > 0 and len(coefs_files):
                if b0 and samples is not None:
                    new_var = (b0 + samples*var/2) / (b0 * lmb0 + samples/2)
                new_flat = ag.util.wavelet.smart_flatten(new_var[digit,mixture,axis])
                plt.semilogy(1/new_flat, label="Posterior"+add)

        plt.semilogy(1/var_flat, label="ML"+add)
    plt.legend(loc=0)
    plt.show()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot posterior variance')
    parser.add_argument('coefficients', nargs='+', metavar='<coef file>', type=argparse.FileType('rb'), help='Filename of model coefficients')
    parser.add_argument('-l', dest='eta', metavar='PENALTY', nargs=1, default=[0.0], type=float, help='Eta for the lambda of the Gamma distribution')
    parser.add_argument('--rho', dest='rho', metavar='RHO', nargs=1, default=[0.0], type=float, help='Rho for the lambda of the Gamma distribution') 
    parser.add_argument('-b', metavar='B', nargs=1, default=[0.0], type=float, help='Prior hypercoefficient b of Gamma distribution')
    parser.add_argument('-i', '--index', nargs=3, metavar=('DIGIT', 'MIXTURE', 'AXIS'), default=(0, 0, 0), type=int, help='Index of data, with choice of DIGIT, MIXTURE and AXIS. Defaults to (0, 0, 0).')

    args = parser.parse_args()
    coef_files = args.coefficients
    eta = args.eta[0]
    rho = args.rho[0]
    b0 = args.b[0]
    digit, mixture, axis = args.index

    plotb(coef_files, eta, rho, b0, digit, mixture, axis)
