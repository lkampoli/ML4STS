ifort -c forpy_mod.F90
ifort -pg skl2f.F90 forpy_mod.o `python3-config --ldflags` -o skl2f.x
./skl2f.x
