What's happening here?
======================

A communication interface between a simple Fortran caller and a python callee is created.
In this way, it is possible to predict transport coefficients with a pre-trained MLP from a CFD solver.

1. In `save_eta.py` a MLP is trained and saved in `mlp.joblib`
2. In `load_eta.py` the trained MLP is loaded and the output dumped in `out`
3. By compiling call_eta.f90 with

~~~~~~~~~
gfortran call_eta.f90 -o call_eta
~~~~~~~~~

and executing with

~~~~~~~~~
./call_eta
~~~~~~~~~

will run the following python command

~~~~~~~~~
python3 load_eta.py 0.2 5000
~~~~~~~~~

and read/write the output to screen.

@TODO:

* Speed-up the communication possibly avoiding read/write phase
* Avoid database loading at each prediction -- STATUS: done!
