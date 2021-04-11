#!/usr/bin/env python
# coding: utf-8

# https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/

import time
import sys
sys.path.insert(0, '../../../../../Utilities/')

from plotting import newfig, savefig
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import pandas as pd
import seaborn as sns

import operator
import itertools

from sklearn import metrics
from sklearn.metrics import *

from sklearn.preprocessing import StandardScaler

from sklearn import model_selection
from sklearn.model_selection import train_test_split, GridSearchCV

import joblib
from joblib import dump, load
import pickle

from sklearn.inspection import permutation_importance

n_jobs = 2

# Import database
with open('../../../../Data/TCs_air5.txt') as f:
    lines = (line for line in f if not line.startswith('#'))
    dataset = np.loadtxt(lines, skiprows=1)

#dataset = np.loadtxt("../../../Data/TCs_air5.txt")
x = dataset[:,0:7] # T, P, x_N2, x_O2, x_NO, x_N, x_O
y = dataset[:,7:8]

# The data is then split into training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=69, shuffle=True)

sc_x = StandardScaler()
sc_y = StandardScaler()

sc_x.fit(x_train)
x_train = sc_x.transform(x_train)
x_test  = sc_x.transform(x_test)

sc_y.fit(y_train)
y_train = sc_y.transform(y_train)
y_test  = sc_y.transform(y_test)

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:',   y_train.shape)
print('Testing Features Shape:',  x_test.shape)
print('Testing Labels Shape:',    y_test.shape)

# Compare Algorithms
from sklearn.tree import DecisionTreeRegressor
from sklearn import kernel_ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor
from sklearn import neighbors
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn import svm
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn import ensemble
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor

#est = DecisionTreeRegressor()
#est=ensemble.ExtraTreesRegressor()
#est=kernel_ridge.KernelRidge()
#est=ensemble.RandomForestRegressor()
#est=svm.SVR()
#est=MLPRegressor()
#est=neighbors.KNeighborsRegressor()
#est=ensemble.GradientBoostingRegressor()
#est=ensemble.HistGradientBoostingRegressor()
#est=ensemble.BaggingRegressor()

#regr = GridSearchCV(est, cv=5, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2')

# prepare configuration for cross validation test harness
seed = 69

# prepare models
models = []
models.append(('DT',  DecisionTreeRegressor()))
models.append(('ET',  ExtraTreesRegressor()))
#models.append(('KR',  KernelRidge()))
models.append(('RF',  RandomForestRegressor()))
models.append(('SVR', SVR()))
models.append(('MLP', MLPRegressor()))
models.append(('KN',  KNeighborsRegressor()))
models.append(('GB',  GradientBoostingRegressor()))
models.append(('HGB', HistGradientBoostingRegressor()))
models.append(('B',   BaggingRegressor()))

# evaluate each model in turn
results = []
names = []
#scoring='explained_variance'
scoring='max_error'
#scoring='neg_mean_absolute_error'
#scoring='neg_mean_squared_error'
#scoring='neg_root_mean_squared_error'
#scoring='neg_mean_squared_log_error'
#scoring='neg_median_absolute_error'
#scoring='r2'
#scoring='neg_mean_poisson_deviance'
#scoring='neg_mean_gamma_deviance'
#scoring='neg_mean_absolute_percentage_error'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
	cv_results = model_selection.cross_val_score(model, x_train, y_train.ravel(), cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.savefig("shear_algorithm_comparison.pdf", dpi=150, crop='false')
plt.show()
