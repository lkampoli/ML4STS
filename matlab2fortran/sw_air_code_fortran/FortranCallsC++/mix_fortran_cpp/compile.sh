gfortran -c fortran_matrix_multiply.f90
g++ -c -std=c++11 cpp_main_1.cpp
gfortran fortran_matrix_multiply.o cpp_main_1.o -lstdc++ -o mix_example.out
