# Regression of Transport coefficients for the State-to-State approach

## Useful Links
https://machinelearningmastery.com/multi-output-regression-models-with-python/
https://machinelearningmastery.com/deep-learning-models-for-multi-output-regression/
https://scikit-learn.org/stable/modules/multiclass.html#multiclass-and-multilabel-algorithms

## Changelog
* The name of the regressors have been changed, omitting the MO (for MultiOutput) because
  the majority of the used algoritmhs natively support MultiOutput and so there so no need
  to explicitely invoke it.
* The best solution to deal with bigger-than-RAM datasets is to use [Dask](https://dask.org/),
  even better if used on clusters.

## Coupling CFD and ML
In the present case, we aim at coupling the modern Fortran solver SPARK with the best performing
machine learning pre-trained algorithm.

In order to use ML in CFD solver, it is necessary to develop an interface. Several, option exist:

* execute_command()
* cython
* CFFI
* ...

## Dataset generation
...
