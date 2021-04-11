#!/bin/bash

# Generate database (if necessary) by calling Matlab

# Binary mixtures (N2/N, O2/O)
cd ../data/bi_mix_sts/
echo $PWD
echo "Generating database ...!"
./run_matlab.sh
cd -
echo "Database generated!"

# Air 5 mixture: N2, O2, NO, N, O
cd ../data/sw_air_code/
echo $PWD
echo "Generating database ...!"
./run_matlab.sh
cd -
echo "Database generated!"

# List of processes
declare -a process=("DR" "VT" "VV" "VV2" "ZR" "all")

# List of algorithms
declare -a algorithm=("DT" "ET" "RF" "SVM" "GB" "HGB" "KN" "KR" "MLP")

#for p in "${process[@]}";
#  do
    for a in "${algorithm[@]}";
      do
#       python run_regression.py -p $p -a $a
        python run_regression.py -a $a
      done
#  done
