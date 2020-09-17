#!/bin/bash
cd src
for i in {0..46}; do \
  #python3 regression.py $i
  #python3 regression_dis.py $i
  python3 regression_rec.py $i
done
cd ..
