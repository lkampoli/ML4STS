FC=gfortran
FCFLAGS=-pg
LFLAGS=-ltensorflow
# -I/usr/local/include -L/usr/local/lib hello_tf.c -ltensorflow -o hello_tf
test_TF_Version: ftf.o test_TF_Version.o
	$(FC) $(FCFLAGS) ftf.o test_TF_Version.o -o $@ $(LFLAGS)

%.o: %.f90
	$(FC) $(FCFLAGS) -c $<

%.o: %.F90
	$(FC) $(FCFLAGS) -c $<

%.o: %.f03
	$(FC) $(FCFLAGS) -c $<

%.o: %.F03
	$(FC) $(FCFLAGS) -c $<

clean:
	rm -f *.o *.mod test_TF_Version gmon.out
