#!/bin/sh

# Variables
N_EPOCHS=$1

for ARCH in 1 2
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

#for ARCH in 3 4 5 6 7 8
#do
#  # Extra Experiments - 5 candidates, 40 voters
#  python main.py -f data/euclidean_nC5_nV40_nP100_imC80.profiles -c 5 -v 40 -e $N_EPOCHS -opt Adam -lr 0.001 -tp 0.2 -s 69420 -arch $ARCH
#  python main.py -f data/gaussian_nC5_nV40_nP100_imC80.profiles -c 5 -v 40 -e $N_EPOCHS -opt Adam -lr 0.001 -tp 0.2 -s 69420 -arch $ARCH
#  python main.py -f data/VMFHypercircle_nC5_nV40_nP100_imC80.profiles -c 5 -v 40 -e $N_EPOCHS -opt Adam -lr 0.001 -tp 0.2 -s 69420 -arch $ARCH
#  python main.py -f data/VMFHypersphere_nC5_nV40_nP100_imC80.profiles -c 5 -v 40 -e $N_EPOCHS -opt Adam -lr 0.001 -tp 0.2 -s 69420 -arch $ARCH
#
#  # Extra Experiments - 5 candidates, 80 voters
#  python main.py -f data/euclidean_nC5_nV80_nP100_imC80.profiles -c 5 -v 80 -e $N_EPOCHS -opt Adam -lr 0.001 -tp 0.2 -s 69420 -arch $ARCH
#  python main.py -f data/gaussian_nC5_nV80_nP100_imC80.profiles -c 5 -v 80 -e $N_EPOCHS -opt Adam -lr 0.001 -tp 0.2 -s 69420 -arch $ARCH
#  python main.py -f data/VMFHypercircle_nC5_nV80_nP100_imC80.profiles -c 5 -v 80 -e $N_EPOCHS -opt Adam -lr 0.001 -tp 0.2 -s 69420 -arch $ARCH
#  python main.py -f data/VMFHypersphere_nC5_nV80_nP100_imC80.profiles -c 5 -v 80 -e $N_EPOCHS -opt Adam -lr 0.001 -tp 0.2 -s 69420 -arch $ARCH
#done
