CC = gcc
FC = gfortran
CFLAGS = -I/usr/include/python2.7
LFLAGS = -L/usr/local/lib -lpython2.7 -lm

.PHONY: all clean

all: rootF main

rootF: rootF.o rootC.o
	$(FC) $^ -o $@ $(LFLAGS)

rootF.o: rootF.F90
	$(FC) -c $< -o $@

main: main.o rootC.o
	$(CC) $^ -o $@ $(LFLAGS)

main.o: main.c
	$(CC) $(CFLAGS) -c $< -o $@

rootC.o: rootC.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o
