#!/bin/sh

# Variables
N_EPOCHS=$1

for ARCH in 1 4 5
do
  # Main Experiments - 5 candidates, 40 voters
  python main.py -f data/spheroid_nC5_nV40_nP100_imC80.profiles -c 5 -v 40 -e $N_EPOCHS -opt Adam -lr 0.001 -tp 0.2 -s 69420 -arch $ARCH
  python main.py -f data/cubic_nC5_nV40_nP100_imC80.profiles -c 5 -v 40 -e $N_EPOCHS -opt Adam -lr 0.001 -tp 0.2 -s 69420 -arch $ARCH
  python main.py -f data/ladder_nC5_nV40_nP100_imC80.profiles -c 5 -v 40 -e $N_EPOCHS -opt Adam -lr 0.001 -tp 0.2 -s 69420 -arch $ARCH

  # Main Experiments - 5 candidates, 80 voters
  python main.py -f data/spheroid_nC5_nV80_nP100_imC80.profiles -c 5 -v 80 -e $N_EPOCHS -opt Adam -lr 0.001 -tp 0.2 -s 69420 -arch $ARCH
  python main.py -f data/cubic_nC5_nV80_nP100_imC80.profiles -c 5 -v 80 -e $N_EPOCHS -opt Adam -lr 0.001 -tp 0.2 -s 69420 -arch $ARCH
  python main.py -f data/ladder_nC5_nV80_nP100_imC80.profiles -c 5 -v 80 -e $N_EPOCHS -opt Adam -lr 0.001 -tp 0.2 -s 69420 -arch $ARCH
done
