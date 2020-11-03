#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

import os
import time
import sys
sys.path.insert(0, '../../../../../Utilities/')

from plotting import newfig, savefig
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import operator
import itertools

from sklearn import metrics
from sklearn.metrics import *

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.inspection import permutation_importance

from joblib import dump, load
import pickle

from sklearn.tree import DecisionTreeRegressor

from sklearn.multioutput import MultiOutputRegressor

n_jobs = -1
trial  = 1

with open('../../../../Data/TCs_air5_MD_full_SMALL_T500_P1000.txt') as f:
    lines = (line for line in f if not line.startswith('#'))
    dataset = np.loadtxt(lines, skiprows=1)

#dataset = np.loadtxt("../../../Data/TCs_air5.txt")
x = dataset[:,0:7] # T, P, x_N2, x_O2, x_NO, x_N, x_O
#y = dataset[:,10:11] # Dij[0][0]
y = dataset[:,10:] # Dij[][]

# The data is then split into training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=69)

sc_x = StandardScaler()
sc_y = StandardScaler()

sc_x.fit(x_train)
x_train = sc_x.fit_transform(x_train)
x_test  = sc_x.fit_transform(x_test)

sc_y.fit(y_train)
y_train = sc_y.transform(y_train)
y_test  = sc_y.transform(y_test)

dump(sc_x, open('../../scaler/scaler_x_MD.pkl', 'wb'))
dump(sc_y, open('../../scaler/scaler_y_MD.pkl', 'wb'))

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', x_test.shape)
print('Testing Labels Shape:', y_test.shape)

hyper_params = [{'criterion': ('mse', 'friedman_mse', 'mae'),
                 'splitter': ('best', 'random'),
                 'max_features': ('auto', 'sqrt', 'log2'),
}]

est = DecisionTreeRegressor()
gs  = GridSearchCV(est, cv=10, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2')

t0 = time.time()
gs.fit(x_train, y_train)
runtime = time.time() - t0
print("Complexity and bandwidth selected and model fitted in %.6f s" % runtime)

train_score_mse = mean_squared_error(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_mae = mean_absolute_error(sc_y.inverse_transform(y_train),sc_y.inverse_transform(gs.predict(x_train)))
train_score_evs = explained_variance_score(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_me  = max_error(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_msle = mean_squared_log_error(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))

test_score_mse = mean_squared_error(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_mae = mean_absolute_error(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_evs = explained_variance_score(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_me  = max_error(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_msle = mean_squared_log_error(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_r2  = r2_score(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))

print("The model performance for testing set")
print("--------------------------------------")
print('MAE is {}'.format(test_score_mae))
print('MSE is {}'.format(test_score_mse))
print('EVS is {}'.format(test_score_evs))
print('ME is {}'.format(test_score_me))
print('MSLE is {}'.format(test_score_msle))
print('R2 score is {}'.format(test_score_r2))

sorted_grid_params = sorted(gs.best_params_.items(), key=operator.itemgetter(0))

out_text = '\t'.join(['regression',
                      str(trial),
                      str(sorted_grid_params).replace('\n',','),
                      str(train_score_mse),
                      str(train_score_mae),
                      str(train_score_evs),
                      str(train_score_me),
                      str(train_score_msle),
                      str(test_score_mse),
                      str(test_score_mae),
                      str(test_score_evs),
                      str(test_score_me),
                      str(test_score_msle),
                      str(runtime)])
print(out_text)
sys.stdout.flush()

best_criterion = gs.best_params_['criterion']
best_splitter  = gs.best_params_['splitter']
best_max_features = gs.best_params_['max_features']

outF = open("output_MD.txt", "w")
print('best_criterion = ', best_criterion, file=outF)
print('best_splitter = ', best_splitter, file=outF)
print('best_max_features = ', best_max_features, file=outF)
outF.close()

regr = DecisionTreeRegressor(criterion='mse',
                             splitter='best',
                             max_features='auto',
                             random_state=69)

regr = MultiOutputRegressor(estimator=regr, n_jobs=n_jobs)

t0 = time.time()
regr.fit(x_train, y_train)
regr_fit = time.time() - t0
print("Complexity and bandwidth selected and model fitted in %.6f s" % regr_fit)

t0 = time.time()
y_regr = regr.predict(x_test)
regr_predict = time.time() - t0
print("Prediction for %d inputs in %.6f s" % (x_test.shape[0], regr_predict))

x_test_dim = sc_x.inverse_transform(x_test)
y_test_dim = sc_y.inverse_transform(y_test)
y_regr_dim = sc_y.inverse_transform(y_regr)

plt.scatter(x_test_dim[:,0], y_test_dim[:,0], s=5, c='k', marker='o', label='KAPPA')
plt.scatter(x_test_dim[:,0], y_regr_dim[:,0], s=5, c='r', marker='d', label='k-Nearest Neighbour')
#plt.scatter(x_test_dim[:,0], y_test_dim[:,1], s=5, c='k', marker='o', label='KAPPA')
#plt.scatter(x_test_dim[:,0], y_regr_dim[:,1], s=5, c='r', marker='d', label='k-Nearest Neighbour')
#plt.scatter(x_test_dim[:,0], y_test_dim[:,2], s=5, c='k', marker='o', label='KAPPA')
#plt.scatter(x_test_dim[:,0], y_regr_dim[:,2], s=5, c='r', marker='d', label='k-Nearest Neighbour')
#plt.title('Shear viscosity regression with kNN')
#plt.ylabel(r'$\eta$ [Pa·s]')
plt.ylabel(' ')
plt.xlabel('T [K] ')
plt.legend()
plt.tight_layout()
plt.savefig("../../pdf/regression_MD.pdf", dpi=150, crop='false')
plt.show()

# save the model to disk
dump(regr, '../../model/model_MD.sav')
