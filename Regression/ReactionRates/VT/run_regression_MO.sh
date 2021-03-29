#!/bin/bash

declare -a dataset_down=("VT_RATES-N2-N2-vt_down"
                         "VT_RATES-N2-O2-vt_down"
                         "VT_RATES-N2-NO-vt_down"
                         "VT_RATES-N2-N_-vt_down"
                         "VT_RATES-N2-O_-vt_down"
                         "VT_RATES-O2-N2-vt_down"
                         "VT_RATES-O2-O2-vt_down"
                         "VT_RATES-O2-NO-vt_down"
                         "VT_RATES-O2-N_-vt_down"
                         "VT_RATES-O2-O_-vt_down"
                         "VT_RATES-NO-N2-vt_down"
                         "VT_RATES-NO-O2-vt_down"
                         "VT_RATES-NO-NO-vt_down"
                         "VT_RATES-NO-N_-vt_down"
                         "VT_RATES-NO-O_-vt_down"
                        )

declare -a dataset_up=("VT_RATES-N2-N2-vt_up"
                       "VT_RATES-N2-O2-vt_up"
                       "VT_RATES-N2-NO-vt_up"
                       "VT_RATES-N2-N_-vt_up"
                       "VT_RATES-N2-O_-vt_up"
                       "VT_RATES-O2-N2-vt_up"
                       "VT_RATES-O2-O2-vt_up"
                       "VT_RATES-O2-NO-vt_up"
                       "VT_RATES-O2-N_-vt_up"
                       "VT_RATES-O2-O_-vt_up"
                       "VT_RATES-NO-N2-vt_up"
                       "VT_RATES-NO-O2-vt_up"
                       "VT_RATES-NO-NO-vt_up"
                       "VT_RATES-NO-N_-vt_up"
                       "VT_RATES-NO-O_-vt_up"
                       )

# Machine learning algorithms
declare -a algorithms=("DT" "ET" "RF" "SVM" "GB" "HGB" "KN" "KR" "MLP")

for i in "${algorithms[@]}";
  do
    cd $i; echo "algorithm: " $i
    echo $PWD
    for j in "${dataset_down[@]}";
      do
        python regression_MO.py $j
      done
    for j in "${dataset_up[@]}";
      do
        python regression_MO.py $j
      done
    cd ..
  done
