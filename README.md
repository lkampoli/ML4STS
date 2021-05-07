
Welcome to ML4STS!
==================

Machine Learning for State-to-State

[![binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/lkampoli/ML4STS/HEAD)
[![stars](https://img.shields.io/github/stars/lkampoli/ML4STS?color=yellow&style=plastic)](https://github.com/lkampoli/ML4STS/stargazers)
[![forks](https://img.shields.io/github/forks/lkampoli/ML4STS?style=plastic)](https://github.com/lkampoli/ML4STS/network/members)
[![watchers](https://img.shields.io/github/watchers/lkampoli/ML4STS?color=green&style=plastic)](https://github.com/lkampoli/ML4STS/watchers)
[![license](https://img.shields.io/github/license/lkampoli/ML4STS?color=orange&style=plastic)](https://www.gnu.org/licenses/lgpl-3.0)
[![activity](https://img.shields.io/github/commit-activity/m/lkampoli/ML4STS?color=red&style=plastic)](https://github.com/lkampoli/ML4STS/graphs/commit-activity)
[![size](https://img.shields.io/github/languages/code-size/lkampoli/ML4STS?color=violet&style=plastic)]()
[![lines](https://img.shields.io/tokei/lines/github/lkampoli/ML4STS?color=pink&style=plastic)]()

This project aims at investigating the usage of machine learning
algorithms for the solution of high-speed (viscous and inviscid,
reacting and non-reacting) non-equilibrium flows according to a
state-to-state (STS) formulation. Several machine learning methods,
including neural networks are considered. In this framework,
the following tasks have been identified:

- [x] Regression of transport coefficients
- [x] Regression of chemical reaction rates
- [x] Regression of chemical relaxation terms
- [x] Coupling of machine learning with ODE solver
- [x] Coupling of machine learning with PDE solver
- [x] Solution of Euler and Navier-Stokes equations with NN
- [ ] RNN/LSTM for ODE integration
- [ ] CNN for solution inference
- [ ] GAN for data generation and super-resolution

## Requisites
Assuming an available python3 version, the following 
packages may be required in order to run some tasks:

* [scikit-learn 0.23](https://scikit-learn.org/stable/)
* [keras](https://keras.io/)
* [tensorflow 1.14.0](https://www.tensorflow.org/)
* [pytorch](https://pytorch.org/)
* [dask](https://dask.org/)
* [matplotlib](https://matplotlib.org/)
* [pandas](https://pandas.pydata.org/)
* [scipy](https://www.scipy.org/)

## Nomenclature
In the directory tree, the following abbreviations have been used:

* DT  - Decision Tree
* SVM - Support Vector Machine
* KR  - Kernel Ridge
* LDA - Linear Discriminant Analysis
* QDA - Quadratic Discriminant Analysis
* PCA - Principal Component Analysis
* SVD - Singular Value Decomposition
* kNN - Nearest Neighbors
* GP  - Gaussian Processes
* NB  - Naive Bayes
* RF  - Random Forest
* ET  - Extreme Tree
* GB  - Gradient Boosting
* HGB - Histogram-Based Gradient Boosting
* MLP - Multi-layer Perceptron
* NN  - Neural Network

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

## Coupling of machine learning with ODE solver
- [x] model implementation
- [ ] find optimal coupling strategy

## Coupling of machine learning with PDE solver
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

<!-- ## Useful Links
     https://machinelearningmastery.com/multi-output-regression-models-with-python/
     https://machinelearningmastery.com/deep-learning-models-for-multi-output-regression/
     https://scikit-learn.org/stable/modules/multiclass.html#multiclass-and-multilabel-algorithms -->

## Copyrights
ML4STS is an open source project, it is distributed under the
[LGPL v3](https://www.gnu.org/licenses/lgpl-3.0.en.html). Anyone interested in
using, developing or contributing to ML4STS is welcome. Take a look at the
[contributing guidelines](CONTRIBUTING.md) to start to contribute to the project.

## Citing ML4STS
If you use ML4STS in your scientific publications, we would appreciate citations to the following paper:

* Campoli, L., Kustova, E., & Maltseva, P. (2021). [Assessment of machine learning methods for state-to-state approaches.](https://arxiv.org/pdf/2104.01042.pdf) arXiv preprint arXiv:2104.01042.

**Bibtex**
```bibtex
@article{campoli2021assessment,
 title={Assessment of machine learning methods for state-to-state approaches},
 author={Campoli, Lorenzo and Kustova, Elena and Maltseva, Polina},
 journal={arXiv preprint arXiv:2104.01042},
 year={2021}
}

## Bibliography
[1] Nagnibeda, E., & Kustova, E. (2009). Non-equilibrium reacting gas flows: kinetic theory of transport and relaxation processes. Springer Science & Business Media.

[2] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Vanderplas, J. (2011). Scikit-learn: Machine learning in Python. the Journal of machine Learning research, 12, 2825-2830.

[3] Géron, A. (2019). Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow: Concepts, tools, and techniques to build intelligent systems. O'Reilly Media.

[4] Chollet, F. (2017). Deep Learning with Python.

[5] VanderPlas, J. (2016). Python data science handbook: Essential tools for working with data. " O'Reilly Media, Inc.".

[6] Mao, Z., Jagtap, A. D., & Karniadakis, G. E. (2020). Physics-informed neural networks for high-speed flows. Computer Methods in Applied Mechanics and Engineering, 360, 112789.
