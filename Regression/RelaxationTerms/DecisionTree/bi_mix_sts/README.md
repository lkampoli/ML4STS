# 14/10/20:

Today, I tried to couple matlab and a machine learning regressor to solve the relaxation of a binary mixture of N2/N
after a steady one-dimensional shock-wave.

To me, it is not yet clear what to regress, in fact, there are several possibilities, differntly convenient depending of the aim,
for example:

* regression of the full RHS after the inversion of the AX=B system; In this case the advantage is that we do not loose time
  to invert it numerically.

* regression of the RHS, only the contribution of one or several processes to the source term, before the inversion of the system.
* regression of the reaction rate coefficients, (k_ci).
