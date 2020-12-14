gcc -c main.c
gfortran -c intrealstring.f90
gfortran main.o intrealstring.o -o cwrapper
