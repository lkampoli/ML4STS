#  to build the shared library libplugin.dylib we execute
python3 builder.py

#we can compile the Fortran program using the following command:
#gfortran -o test -L./ -lplugin test.f90
ifort -o test -L./ -lplugin test.f90
./test

