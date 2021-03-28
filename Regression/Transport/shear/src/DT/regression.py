#!/usr/bin/env python
# coding: utf-8

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

from sklearn.model_selection import train_test_split, GridSearchCV

import joblib
from joblib import dump, load
import pickle

from sklearn.inspection import permutation_importance

from sklearn.tree import DecisionTreeRegressor

n_jobs = -1
trial  = 1

# Import database
with open('../../../../Data/TCs_air5.txt') as f:
    lines = (line for line in f if not line.startswith('#'))
    dataset = np.loadtxt(lines, skiprows=1)

print(dataset.shape)

x = dataset[:,0:7] # T, P, x_N2, x_O2, x_NO, x_N, x_O
y = dataset[:,7:8] # shear

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=69)

sc_x = StandardScaler()
sc_y = StandardScaler()

sc_x.fit(x_train)
x_train = sc_x.transform(x_train)
x_test  = sc_x.transform(x_test)

sc_y.fit(y_train)
y_train = sc_y.transform(y_train)
y_test  = sc_y.transform(y_test)

dump(sc_x, open('scaler_x_shear.pkl', 'wb'))
dump(sc_y, open('scaler_y_shear.pkl', 'wb'))

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:',   y_train.shape)
print('Testing Features Shape:',  x_test.shape)
print('Testing Labels Shape:',    y_test.shape)

hyper_params = [{'criterion': ('mse', 'friedman_mse', 'mae'), # mse
                 'splitter': ('best', 'random'),              # auto
                 'max_features': ('auto', 'sqrt', 'log2'),    # best
}]

est = DecisionTreeRegressor()
gs  = GridSearchCV(est, cv=10, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2')

t0 = time.time()
gs.fit(x_train, y_train.ravel())
runtime = time.time() - t0
print("Complexity and bandwidth selected and model fitted in %.6f s" % runtime)

train_score_mse  = mean_squared_error(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_mae  = mean_absolute_error(sc_y.inverse_transform(y_train),sc_y.inverse_transform(gs.predict(x_train)))
train_score_evs  = explained_variance_score(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_me   = max_error(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_msle = mean_squared_log_error(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_r2   = r2_score(                sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))

test_score_mse  = mean_squared_error(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_mae  = mean_absolute_error(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_evs  = explained_variance_score(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_me   = max_error(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_msle = mean_squared_log_error(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_r2   = r2_score(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))

print()
print("The model performance for training set")
print("--------------------------------------")
print('MAE is      {}'.format(train_score_mae))
print('MSE is      {}'.format(train_score_mse))
print('EVS is      {}'.format(train_score_evs))
print('ME is       {}'.format(train_score_me))
print('MSLE is     {}'.format(train_score_msle))
print('R2 score is {}'.format(train_score_r2))
print()
print("The model performance for testing set")
print("--------------------------------------")
print('MAE is      {}'.format(test_score_mae))
print('MSE is      {}'.format(test_score_mse))
print('EVS is      {}'.format(test_score_evs))
print('ME is       {}'.format(test_score_me))
print('MSLE is     {}'.format(test_score_msle))
print('R2 score is {}'.format(test_score_r2))
print()
print("Best parameters set found on development set:")
print(gs.best_params_)
print()

# Re-train with best parameters
regr = DecisionTreeRegressor(**gs.best_params_)

t0 = time.time()
regr.fit(x_train, y_train.ravel())
regr_fit = time.time() - t0
print("Complexity and bandwidth selected and model fitted in %.6f s" % regr_fit)

importance = regr.feature_importances_

# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))

# plot feature importance
plt.title("Feature importances")
features = np.array(['T', 'P', '$X_{N2}$', '$X_{O2}$', '$X_{NO}$', '$X_N$', '$X_O$'])
plt.bar(features, importance)
#plt.bar([x for x in range(len(importance))], importance)
plt.savefig("importance_shear.pdf", dpi=150, crop='false')
plt.show()
plt.close()

t0 = time.time()
y_regr = regr.predict(x_test)
regr_predict = time.time() - t0
print("Prediction for %d inputs in %.6f s" % (x_test.shape[0], regr_predict))

with open('output.log', 'w') as f:
    print("Training time: %.6f s" % regr_fit, file=f)
    print("Prediction time: %.6f s" % regr_predict, file=f)
    print(" ", file=f)
    print("The model performance for training set", file=f)
    print("--------------------------------------", file=f)
    print('MAE is {}'.format(train_score_mae), file=f)
    print('MSE is {}'.format(train_score_mse), file=f)
    print('EVS is {}'.format(train_score_evs), file=f)
    print('ME is {}'.format(train_score_me), file=f)
    print('MSLE is {}'.format(train_score_msle), file=f)
    print('R2 score is {}'.format(train_score_r2), file=f)
    print(" ", file=f)
    print("The model performance for testing set", file=f)
    print("--------------------------------------", file=f)
    print('MAE is {}'.format(test_score_mae), file=f)
    print('MSE is {}'.format(test_score_mse), file=f)
    print('EVS is {}'.format(test_score_evs), file=f)
    print('ME is {}'.format(test_score_me), file=f)
    print('MSLE is {}'.format(test_score_msle), file=f)
    print('R2 score is {}'.format(test_score_r2), file=f)
    print(" ", file=f)
    print("Best parameters set found on development set:", file=f)
    print(gs.best_params_, file=f)

x_test_dim = sc_x.inverse_transform(x_test)
y_test_dim = sc_y.inverse_transform(y_test)
y_regr_dim = sc_y.inverse_transform(y_regr)

plt.scatter(x_test_dim[:,0], y_test_dim[:], s=5, c='k', marker='o', label='KAPPA')
plt.scatter(x_test_dim[:,0], y_regr_dim[:], s=2.5, c='r', marker='o', label='DecisionTree')
plt.ylabel(r'$\eta$ [Pa·s]')
plt.xlabel('T [K] ')
plt.legend()
plt.tight_layout()
plt.savefig("shear.pdf", dpi=150, crop='false')
plt.show()
plt.close()

# save the model to disk
dump(gs, 'model_shear.sav')
