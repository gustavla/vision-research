from __future__ import division, print_function
import argparse

parser = argparse.ArgumentParser(description='Generate diagnostic images')
parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of model file') 
parser.add_argument('result', metavar='<file>', type=argparse.FileType('rb'), help='Filename of results file')


