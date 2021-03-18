#!/bin/bash
#
gfortran -fno-underscoring -c kronrod_test.f90
if [ $? -ne 0 ]; then
  echo "Errors compiling kronrod_test.f90"
  exit
fi
#
g++ -c kronrod.cpp
if [ $? -ne 0 ]; then
  echo "Errors compiling kronrod.cpp"
  exit
fi
#
gfortran kronrod_test.o kronrod.o -lm -lstdc++
if [ $? -ne 0 ]; then
  echo "Errors linking and loading kronrod_test.o + kronrod.o"
  exit
fi
rm kronrod_test.o
rm kronrod.o
#
mv a.out kronrod
./kronrod > kronrod_output.txt
if [ $? -ne 0 ]; then
  echo "Errors running kronrod"
  exit
fi
rm kronrod
#
echo "Test program output written to kronrod_output.txt."
