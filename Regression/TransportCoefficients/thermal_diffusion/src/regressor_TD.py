#!/usr/bin/env python
# coding: utf-8

import dask
from dask import delayed
from dask.distributed import Client, progress
import dask.dataframe as dd
import dask.bag as db
from dask_ml.wrappers import ParallelPostFit
import dask.array as da
import dask_ml.datasets
import dask_ml.cluster

client = Client(processes=False, threads_per_worker=2, n_workers=2, memory_limit='2GB')
client

import time
import sys
sys.path.insert(0, '../../../../Utilities/')

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

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

import joblib
from joblib import dump, load
import pickle

from sklearn.inspection import permutation_importance

from sklearn.tree import DecisionTreeRegressor

from sklearn.multioutput import MultiOutputRegressor
from sklearn.multioutput import RegressorChain

n_jobs = -1
trial  = 1

# Import database
# https://pandas.pydata.org/pandas-docs/version/0.25.1/reference/api/pandas.DataFrame.to_numpy.html#pandas.DataFrame.to_numpy
#data = pd.read_fwf("../../../Data/TCs_air5.txt").to_numpy()
#x    = data[:,0:7]
#y    = data[:,7:]

# https://stackoverflow.com/questions/17151210/numpy-loadtxt-skip-first-row
with open('../../../Data/TCs_air5.txt') as f:
    lines = (line for line in f if not line.startswith('#'))
    dataset = np.loadtxt(lines, skiprows=1)

#dataset = np.loadtxt("../../../Data/TCs_air5.txt")
x = dataset[:,0:7] # T, P, x_N2, x_O2, x_NO, x_N, x_O
y = dataset[:,7:]  # D_Ti

# The data is then split into training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=69)

sc_x = StandardScaler()
sc_y = StandardScaler()

sc_x.fit(x_train)
x_train = sc_x.transform(x_train)
x_test  = sc_x.transform(x_test)

sc_y.fit(y_train)
y_train = sc_y.transform(y_train)
y_test  = sc_y.transform(y_test)

dump(sc_x, open('../scaler/scaler_x_TD.pkl', 'wb'))
dump(sc_y, open('../scaler/scaler_y_TD.pkl', 'wb'))

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:',   y_train.shape)
print('Testing Features Shape:',  x_test.shape)
print('Testing Labels Shape:',    y_test.shape)

hyper_params = [{'criterion': ('mse', 'friedman_mse', 'mae'),
                 'splitter': ('best', 'random'),
                 'max_features': ('auto', 'sqrt', 'log2'),
}]

est = DecisionTreeRegressor()
gs  = GridSearchCV(est, cv=5, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2')
#gs  = MultiOutputRegressor(gs)#.fit(X_train, y_train)

# evaluate the model and collect the scores
#n_scores = cross_val_score(est, x_train, y_train, scoring='neg_mean_absolute_error', cv=10, n_jobs=-1)

# force the scores to be positive
#n_scores = np.absolute(n_scores)

# summarize performance
#print('MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
#print("Accuracy: %0.2f (+/- %0.2f)" % (n_scores.mean(), n_scores.std() * 2))

t0 = time.time()
#with joblib.parallel_backend('dask'):
    #gs.fit(x_train, y_train.ravel())
gs.fit(x_train, y_train)
runtime = time.time() - t0
print("Complexity and bandwidth selected and model fitted in %.6f s" % runtime)

# https://stackoverflow.com/questions/60942564/how-to-grid-search-parameter-for-xgboost-with-multioutputregressor-wrapper
#print(best_params = gs.estimators_[0].best_params_)  # for the first y_target estimator

train_score_mse = mean_squared_error(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_mae = mean_absolute_error(sc_y.inverse_transform(y_train),sc_y.inverse_transform(gs.predict(x_train)))
train_score_evs = explained_variance_score(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_me  = max_error(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
#train_score_msle = mean_squared_log_error(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))

test_score_mse = mean_squared_error(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_mae = mean_absolute_error(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_evs = explained_variance_score(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_me  = max_error(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
#test_score_msle = mean_squared_log_error(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_r2  = r2_score(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))

print("The model performance for testing set")
print("--------------------------------------")
print('MAE is {}'.format(test_score_mae))
print('MSE is {}'.format(test_score_mse))
print('EVS is {}'.format(test_score_evs))
print('ME is {}'.format(test_score_me))
print('R2 score is {}'.format(test_score_r2))

sorted_grid_params = sorted(gs.best_params_.items(), key=operator.itemgetter(0))

out_text = '\t'.join(['regression',
                      str(trial),
                      str(sorted_grid_params).replace('\n',','),
                      str(train_score_mse),
                      str(train_score_mae),
                      str(train_score_evs),
                      str(train_score_me),
#                     str(train_score_msle),
                      str(test_score_mse),
                      str(test_score_mae),
                      str(test_score_evs),
                      str(test_score_me),
#                     str(test_score_msle),
                      str(runtime)])
print(out_text)
sys.stdout.flush()

best_criterion = gs.best_params_['criterion']
best_splitter  = gs.best_params_['splitter']
best_max_features = gs.best_params_['max_features']

outF = open("output_TD.txt", "w")
print('best_criterion = ', best_criterion, file=outF)
print('best_splitter = ', best_splitter, file=outF)
print('best_max_features = ', best_max_features, file=outF)
outF.close()

regr = DecisionTreeRegressor(criterion=best_criterion,
                             splitter=best_splitter,
                             max_features=best_max_features,
                             random_state=69)

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
plt.savefig("../pdf/importance_TD.pdf", dpi=150, crop='false')
plt.show()
plt.close()

t0 = time.time()
y_regr = regr.predict(x_test)
regr_predict = time.time() - t0
print("Prediction for %d inputs in %.6f s" % (x_test.shape[0], regr_predict))

# open a file to append
outF = open("output_TD.txt", "a")
print("Complexity and bandwidth selected and model fitted in %.6f s" % regr_fit, file=outF)
print("Prediction for %d inputs in %.6f s" % (x_test.shape[0], regr_predict),file=outF)
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_regr), file=outF)
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_regr), file=outF)
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_regr)), file=outF)
outF.close()

print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_regr))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_regr))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_regr)))

x_test_dim = sc_x.inverse_transform(x_test)
y_test_dim = sc_y.inverse_transform(y_test)
y_regr_dim = sc_y.inverse_transform(y_regr)

plt.scatter(x_test_dim[:,0], y_test_dim[:,0], s=5, c='k', marker='o', label='KAPPA')
plt.scatter(x_test_dim[:,0], y_regr_dim[:,0], s=3, c='r', marker='o', label='DecisionTree')
#plt.title('Shear viscosity regression with kNN')
#plt.ylabel(r'$\eta$ [Pa·s]')
plt.xlabel('T [K] ')
plt.legend()
plt.tight_layout()
plt.savefig("../pdf/regression_TD.pdf", dpi=150, crop='false')
plt.show()
plt.close()

# save the model to disk
dump(gs, '../model/model_TD.sav')
