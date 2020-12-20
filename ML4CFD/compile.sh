gfortran -c ftf.F03 test_TF_Version.F03
gfortran *.o -lftf -o test_TF_Version.x
./test_TF_Version.x
