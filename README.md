# :warning: :construction: Repository under construction

# Machine Learning for State-to-State
This project aims at investigating the usage of machine learning algorithms (MLAs) for the solution of high-speed (viscous and inviscid, reacting and non-reacting) non-equilibrium flows according to the state-to-state (STS) formulation. Several machine learning methods, including neural networks (NNs) will be considered. In this framework, the following tasks have been identified:

* Regression of transport coefficients
* Regression of chemical reaction rate coefficients
* Regression of chemical relaxation terms
* Coupling of machine learning with CFD solver
* Solution of Euler and Navier-Stokes equations with NN

# Requisites
Assuming an available python3 version, the following packages are necessary:
* [scikit-learn 0.23](https://scikit-learn.org/stable/)
* [keras](https://keras.io/)
* [tensorflow 1.14.0](https://www.tensorflow.org/)
* [pytorch](https://pytorch.org/)
* [cffi](https://cffi.readthedocs.io/en/latest/)
* [dask](https://dask.org/)
* [matplotlib](https://matplotlib.org/)
* [pandas](https://pandas.pydata.org/)
* [scipy](https://www.scipy.org/)

# Description
Here below a brief description of the repository directories is given.

## Regression of transport coefficients
- [x] model implementation for shear, bulk viscosity, thermal conductivity and thermal/mass diffusion
- [x] hyperparameters tuning
- [ ] solve the problem of big data for the mass diffusion (probably with Dask and/or cluster)
- [ ] coupling with Spark CFD solver
- [ ] define optimal interface Fortran -> Python

## Regression of chemical reaction rate coefficients, k_ci
- [x] model implementation
- [x] hyperparameters tuning

## Regression of chemical relaxation terms, R_ci
- [x] model implementation
- [x] hyperparameters tuning

## Coupling of machine learning with CFD solver
- [x] model implementation
- [ ] find optimal coupling strategy

## Euler_1d_shock_STS
In this directory, I try to solve the full system of Euler equations for a one-dimensional reacting shock flow.
The directory contains both the `.py` and `.ipynb` files, for convenience, but they are the same.
The `PINN.py` deals with the solution of the Euler equations without STS.
The `PINN_STS.py` deals with the solution of the Euler equations with STS.

- [x] model implementation
- [ ] hyperparameters tuning
- [ ] re-write in compact form
- [ ] bigfix

# Useful Links
https://machinelearningmastery.com/multi-output-regression-models-with-python/

https://machinelearningmastery.com/deep-learning-models-for-multi-output-regression/

https://scikit-learn.org/stable/modules/multiclass.html#multiclass-and-multilabel-algorithms

# Copyrights
ML4STS is an open source project, it is distributed under the GPL v3. Anyone is interest to use, to develop or to contribute to ML4STS is welcome.
Take a look at the contributing guidelines for starting to contribute to the project.

# Bibliography
[1] Nagnibeda, E., & Kustova, E. (2009). Non-equilibrium reacting gas flows: kinetic theory of transport and relaxation processes. Springer Science & Business Media.

[2] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Vanderplas, J. (2011). Scikit-learn: Machine learning in Python. the Journal of machine Learning research, 12, 2825-2830.

[3] Géron, A. (2019). Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow: Concepts, tools, and techniques to build intelligent systems. O'Reilly Media.

[4] Chollet, F. (2017). Deep Learning with Python.

[5] VanderPlas, J. (2016). Python data science handbook: Essential tools for working with data. " O'Reilly Media, Inc.".

[6] Mao, Z., Jagtap, A. D., & Karniadakis, G. E. (2020). Physics-informed neural networks for high-speed flows. Computer Methods in Applied Mechanics and Engineering, 360, 112789.
