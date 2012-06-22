import sys
import numpy as np
import pylab as plt
from misc import plot_edges
from amitgroup.io.mnist import read
from amitgroup.features import bedges 

import pstats, cProfile


def main():
    #all_edges = []
    #d = range(9) 
    d = [9]
    images, _ = read('training', '../../mnist', d)
    #images = images[:100]

    all_edges = bedges(images)
    #cProfile.runctx("all_edges = amitedges(images)", globals(), locals(), "Profile.prof")
    #s = pstats.Stats("Profile.prof")
    #s.strip_dirs().sort_stats("time").print_stats()

    print all_edges[0, 0, 0, 0]
    
    platonic_edges = all_edges.mean(axis=0)
    plot_edges(platonic_edges) 
            

if __name__ == '__main__':
    main()

