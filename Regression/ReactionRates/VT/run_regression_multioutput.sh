#!/bin/bash

cd src

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

for i in "${dataset_down[@]}";
do
  python3 regression_multioutput_down.py $i
done

for i in "${dataset_up[@]}";
do
  python3 regression_multioutput_up.py $i
done

cd ..
