# :warning: :construction: Repository under construction

# Machine Learning for State-to-State

This project aims at investigating the usage of machine learning (ML)
algorithms for the solution of high-speed (viscous and inviscid,
reacting and non-reacting) non-equilibrium flows according to a vibrational
state-to-state (STS) formulation. Several machine learning methods,
including neural networks (NN) will be considered. In this framework,
the following tasks have been identified:

* Kinetics
 - Regression of chemical reaction rate coefficients
 - Regression of chemical relaxation terms
* Transport
 - Regression of omega integrals
 - Regression of transport coefficients
* Coupling of ML and CFD
* Solution of Euler equations with DNN

# Requisites

Assuming an available python3 version, the following packages may be required
in order to be able to run some tasks:
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

Here below, a brief description of the repository directories
and state of advancement is given.

## Regression of transport coefficients

- [x] model implementation for shear, bulk viscosity, thermal conductivity and thermal/mass diffusion
- [x] hyperparameters tuning
- [ ] solve the problem of big data for the mass diffusion (probably with Dask and/or cluster, colab)
- [ ] coupling with SPARK CFD solver 
- [ ] define optimal interface Fortran -> Python

## Regression of chemical reaction rate coefficients, `k_ci`

- [x] model implementation
- [x] hyperparameters tuning
- [x] find optimal regressor

## Regression of chemical relaxation terms, `R_ci`

- [x] model implementation
- [x] hyperparameters tuning
- [x] find optimal regressor

## Coupling of ML and CFD

- [x] model implementation
- [ ] find optimal coupling strategy
- [x] coupling of SPARK and ML models implemented in Keras (with [FKB](https://github.com/scientific-computing/FKB)) for Transport (6T)

## `Euler_1d_shock_STS`

In this directory, I try to solve the full system of Euler equations for a one-dimensional reacting shock flow.
The directory contains both the `.py` and `.ipynb` files, for convenience, but they are the same.
The `PINN.py` deals with the solution of the Euler equations without STS.
The `PINN_STS.py` deals with the solution of the Euler equations with STS.

- [x] model implementation
- [ ] hyperparameters tuning
- [x] re-write in compact form
- [ ] bugfix

# Useful Links

Here is an Overleaf page where I collect useful resources about artificial intelligence, 
machine learning and deep learning subdivided in courses, books, papers, repositories, 
web-pages and videos organized by topics.

# Copyrights

ML4STS is an open source project, it is distributed under the GPL v3.
Anyone is interest to use, to develop or to contribute to ML4STS is
welcome. Take a look at the contributing guidelines for starting to contribute to the project.

# Bibliography

[1] Nagnibeda, E., & Kustova, E. (2009). Non-equilibrium reacting gas flows: kinetic theory of transport and relaxation processes. Springer Science & Business Media.

[2] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Vanderplas, J. (2011). Scikit-learn: Machine learning in Python. the Journal of machine Learning research, 12, 2825-2830.

[3] Géron, A. (2019). Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow: Concepts, tools, and techniques to build intelligent systems. O'Reilly Media.

[4] Chollet, F. (2017). Deep Learning with Python.

[5] VanderPlas, J. (2016). Python data science handbook: Essential tools for working with data. " O'Reilly Media, Inc.".

[6] Mao, Z., Jagtap, A. D., & Karniadakis, G. E. (2020). Physics-informed neural networks for high-speed flows. Computer Methods in Applied Mechanics and Engineering, 360, 112789.
