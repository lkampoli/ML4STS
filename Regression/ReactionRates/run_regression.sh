#!/bin/bash

# List of processes
declare -a process=("DR" "VT" "VV" "VV2" "ZR")

# List of algorithms
declare -a algorithm=("DT" "ET" "RF" "SVM" "GB" "HGB" "KN" "KR" "MLP")

for i in "${process[@]}";
  do
    #echo "process: " $i
    for j in "${algorithm[@]}";
      do
        #echo "algorithm: " $j
        python run_regression.py -p $i -a $j
      done
  done
