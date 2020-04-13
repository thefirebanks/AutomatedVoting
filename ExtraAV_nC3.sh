#!/bin/sh

# Variables
N_EPOCHS=$1

for ARCH in 1 2
do
  # Main Experiments - 3 candidates, 20 voters
  python main.py -f data/spheroid_nC3_nV20_nP100_imC40.profiles -c 3 -v 20 -e $N_EPOCHS -opt Adam -lr 0.001 -tp 0.2 -s 69420 -arch $ARCH
  python main.py -f data/cubic_nC3_nV20_nP100_imC40.profiles -c 3 -v 20 -e $N_EPOCHS -opt Adam -lr 0.001 -tp 0.2 -s 69420 -arch $ARCH
  python main.py -f data/ladder_nC3_nV20_nP100_imC40.profiles -c 3 -v 20 -e $N_EPOCHS -opt Adam -lr 0.001 -tp 0.2 -s 69420 -arch $ARCH
done