#!/bin/bash

# Dissociation processes
declare -a dataset_dis=("DR_RATES-N2-N2-dis"
                        "DR_RATES-N2-O2-dis"
                        "DR_RATES-N2-NO-dis"
                        "DR_RATES-N2-N_-dis"
                        "DR_RATES-N2-O_-dis"
                        "DR_RATES-NO-N2-dis"
                        "DR_RATES-NO-O2-dis"
                        "DR_RATES-NO-NO-dis"
                        "DR_RATES-NO-N_-dis"
                        "DR_RATES-O2-N2-dis"
                        "DR_RATES-O2-O2-dis"
                        "DR_RATES-O2-NO-dis"
                        "DR_RATES-O2-N_-dis"
                        "DR_RATES-O2-O_-dis"
                       )

# Recombination processes
declare -a dataset_rec=("DR_RATES-N2-N2-rec"
                        "DR_RATES-N2-O2-rec"
                        "DR_RATES-N2-NO-rec"
                        "DR_RATES-N2-N_-rec"
                        "DR_RATES-N2-O_-rec"
                        "DR_RATES-NO-N2-rec"
                        "DR_RATES-NO-O2-rec"
                        "DR_RATES-NO-NO-rec"
                        "DR_RATES-NO-N_-rec"
                        "DR_RATES-O2-N2-rec"
                        "DR_RATES-O2-O2-rec"
                        "DR_RATES-O2-NO-rec"
                        "DR_RATES-O2-N_-rec"
                        "DR_RATES-O2-O_-rec"
                       )

# Machine learning algorithms
declare -a algorithms=("DT" "ET" "RF" "SVM" "GB" "HGB" "KN" "KR" "MLP")

# Dissociation processes
for i in "${algorithms[@]}";
  do
    cd $i; echo "algorithm: " $i
    echo $PWD
    for j in "${dataset_dis[@]}";
      do
        python regression_MO.py $j
      done
    for j in "${dataset_rec[@]}";
      do
        python regression_MO.py $j
      done
    cd ..
  done

# Recombination processes
#for i in "${algorithms[@]}";
#  do
#    cd $i; echo "algorithm: " $i
#    echo $PWD
#    for j in "${dataset_rec[@]}";
#      do
#        python regression_MO.py $j
#      done
#    cd ..
#  done
