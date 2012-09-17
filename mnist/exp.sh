#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 KERNEL RADIUS"
    exit 1   
fi

KERNEL=$1
RADIUS=$2
DIR="d-$KERNEL-$RADIUS"

rm -rf $DIR
mkdir $DIR

echo "Extracting training features..."
python extract_training_features.py -k 5 -r 0 100 --save-originals --kernel $KERNEL --radius $RADIUS training $DIR/training-feat.npz

echo "Extracting testing featuers..."
python extract_testing_features.py -k 5 -r 2000 7000 --save-originals --kernel $KERNEL --radius $RADIUS training $DIR/testing-feat.npz

echo "Running mixture model..."
python train_mixture.py -e 0.05 -s 0 $DIR/training-feat.npz $DIR/mixtures.npz 5

echo "Train deform coefficients..."
python train_def_coefs.py -l 100 --rho 1.0 -n 12 -b 0.0005 $DIR/training-feat.npz $DIR/mixtures.npz $DIR/coefs.npz

echo "Run classifier..."
python run_classifier.py -d edges -a 1.4 $DIR/testing-feat.npz $DIR/mixtures.npz $DIR/coefs.npz
