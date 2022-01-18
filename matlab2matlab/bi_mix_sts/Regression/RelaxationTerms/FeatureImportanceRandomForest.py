# https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html?highlight=permutation%20accuracy%20importance

import time
import sys
sys.path.insert(0, './')

#from plotting import newfig, savefig
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
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score

from sklearn.inspection import permutation_importance

from sklearn import ensemble
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.decomposition import PCA, KernelPCA, FastICA

from joblib import dump, load
import pickle

n_jobs = 1

with open('../../data/dataset_N2N_rhs.dat.OK') as f:
    lines = (line for line in f if not line.startswith('#'))
    data = np.loadtxt(lines, skiprows=0)

x = data[:,0:56]   # x_s, time_s, Temp, ni_n, na_n, rho, v, p, E, H
y = data[:,56:57]  # rhs[0:50]

print(data.shape)
print("x=",x.shape)
print("y=",y.shape)

#plt.scatter(x[:,1], x[:,10], s=5, c='k', marker='o', label='treanor-marrone')
#plt.ylabel(r'$\eta$ [Pa·s]')
#ax.xlabel('T [K] ')
#plt.yscale('log')
#plt.xscale('log')
#plt.legend()
#plt.show()
#plt.close()

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=69)

print("x=",x_train.shape)
print("y=",y_train.shape)

#sc_x = StandardScaler()
#sc_y = StandardScaler()
#
## fit scaler
#sc_x.fit(x_train)
#
## transform training dataset
#x_train = sc_x.transform(x_train)
#
## transform test dataset
#x_test = sc_x.transform(x_test)
#
## fit scaler on training dataset
#sc_y.fit(y_train)
##sc_y.fit(y_train.reshape(-1,1))
#
## transform training dataset
#y_train = sc_y.transform(y_train)
#
## transform test dataset
#y_test = sc_y.transform(y_test)
#
#print('Training Features Shape:', x_train.shape)
#print('Training Labels Shape:', y_train.shape)
#print('Testing Features Shape:', x_test.shape)
#print('Testing Labels Shape:', y_test.shape)

# Feature importance based on mean decrease in impurity
# -----------------------------------------------------
feature_names = [f"feature {i}" for i in range(x.shape[1])]
print(feature_names)
#est = ensemble.RandomForestRegressor(random_state=69, verbose=2)
est = ensemble.ExtraTreesRegressor(random_state=69, verbose=2)
est.fit(x_train, y_train.ravel())

start_time = time.time()
importances = est.feature_importances_
std = np.std([tree.feature_importances_ for tree in est.estimators_], axis=0)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

# Let's plot the impurity-based importance.
forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

# Feature importance based on feature permutation
# -----------------------------------------------
# Permutation feature importance overcomes limitations of the impurity-based
# feature importance: they do not have a bias toward high-cardinality features
# and can be computed on a left-out test set.

start_time = time.time()
result = permutation_importance(est, x_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
#results = permutation_importance(regr, x_train, y_train, scoring='neg_mean_squared_error')
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

forest_importances = pd.Series(result.importances_mean, index=feature_names)
#importance = results.importances_mean

# The computation for full permutation importance is more costly. Features are
# shuffled n times and the model refitted to estimate the importance of it.
# Please see :ref:`permutation_importance` for more details. We can now plot
# the importance ranking.

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()

print("importances_mea = ", result.importances_mean)
print("importances_std = ", result.importances_std)
