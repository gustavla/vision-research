
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='Test some shit')

parser.add_argument('-k', nargs=1, default=[5], choices=range(1, 7), type=int, help='Sensitivity of features. 1-6, with 6 being the most conservative.')
parser.add_argument('--radius', metavar='RADIUS', nargs=1, default=[1], type=int, help='Inflation radius')
parser.add_argument('--kernel', metavar='KERNEL', nargs=1, default=['box'], type=str, choices=('box', 'along'), help='Kernel shape of inflation')

args = parser.parse_args()
k = args.k[0]
inflation_radius = args.radius[0]
inflation_type = args.kernel[0]

import amitgroup as ag
import warnings
warnings.simplefilter("ignore", np.ComplexWarning)

digit = ag.io.load_mnist('training', return_labels=False, selection=slice(1))[0]

#x, y = np.mgrid[0:1:32j, 0:1:32j]
#digit = np.zeros((32, 32))
#digit = np.exp((-(x-0.5)**2-(y-0.5)**2)*10.0) 

feats = ag.features.bedges(digit, k=k, inflate=inflation_type, radius=inflation_radius)

print feats.shape

ag.plot.images([digit] + list(feats))
