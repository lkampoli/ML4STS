#!/bin/bash

cd src

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

for i in "${dataset_dis[@]}";
do
  python3 regression_multioutput_dis.py $i
done

for i in "${dataset_rec[@]}";
do
  python3 regression_multioutput_rec.py $i
done

cd ..
