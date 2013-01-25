
import argparse
import matplotlib.pylab as plt
import amitgroup as ag
import gv
import numpy as np
from config import VOCSETTINGS
from histogram_of_detections import calc_llhs

def main():
    parser = argparse.ArgumentParser(description='Train mixture model on edge data')
    parser.add_argument('model', metavar='<model file>', type=argparse.FileType('rb'), help='Filename of the model file')
    parser.add_argument('mixcomp', metavar='<mixture component>', type=int, help='mix comp')
    parser.add_argument('output', metavar='<output file>', type=argparse.FileType('wb'), help='Filename of the output file')

    args = parser.parse_args()
    model_file = args.model
    mixcomp = args.mixcomp
    output = args.output

    # Load detector
    detector = gv.Detector.load(model_file)

    print 'Processing positives...'
    llhs_positives = calc_llhs(VOCSETTINGS, detector, True, mixcomp)
    print 'Processing negatives...'
    #llhs_negatives = calc_llhs(VOCSETTINGS, detector, False, mixcomp)

    score = llhs_positives.mean() - llhs_negatives.mean() 

    print '-----'
    print 'Score:', score
    
    np.save(output, dict(llhs_positives=llhs_positives, llhs_negatives=llhs_negatives, score=score))


if __name__ == '__main__':
    main()
