
def plotb(lmb0, b0, digit=0, mixture=0, axis=0):
    import numpy as np
    import matplotlib.pylab as plt
    import amitgroup as ag

    coefs_data = np.load('c-coefs2.npz')
    var = coefs_data['prior_var']
    samples = coefs_data['samples']

    if b0 and lmb0 and samples is not None:
        new_var = (b0 + samples*var/2) / (b0 * lmb0 + samples/2)

    last_i = var.shape[3]-1
    plt.xlim((0, last_i))
    plt.semilogy([0, last_i], [lmb0]*2, label="Prior")
    plt.semilogy(1/var[digit,mixture,axis], label="ML")
    plt.semilogy(1/new_var[digit,mixture,axis], label="Posterior")
    plt.legend(loc=0)
    plt.show()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot posterior variance')
    parser.add_argument('lmb', metavar='LAMBDA', type=float, help='Prior lambda peak of Gamma distribution')
    parser.add_argument('b', metavar='B', type=float, help='Prior hypercoefficient b of Gamma distribution')
    parser.add_argument('-i', '--index', nargs=3, metavar=('DIGIT', 'MIXTURE', 'AXIS'), default=(0, 0, 0), type=int, help='Index of data, with choice of DIGIT, MIXTURE and AXIS. Defaults to (0, 0, 0).')

    args = parser.parse_args()
    lmb0 = args.lmb
    b0 = args.b
    digit, mixture, axis = args.index

    plotb(lmb0, b0, digit, mixture, axis)
