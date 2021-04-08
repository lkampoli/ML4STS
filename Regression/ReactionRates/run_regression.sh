#!/bin/bash

# List of processes
declare -a process=("DR" "VT" "VV" "VV2" "ZR")

# List of algorithms
declare -a algorithm=("DT" "ET" "RF" "SVM" "GB" "HGB" "KN" "KR" "MLP")

for p in "${process[@]}";
  do
    for a in "${algorithm[@]}";
      do
        python run_regression.py -p $p -a $a
      done
  done
