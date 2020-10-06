#!/bin/bash
cd src
for i in {0..46}; do \
  #python3 regression_down.py $i
  python3 regression_up.py $i
done
cd ..
