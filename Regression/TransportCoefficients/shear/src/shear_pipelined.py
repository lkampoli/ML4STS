#!/usr/bin/env python
# coding: utf-8

# https://www.kdnuggets.com/2018/01/managing-machine-learning-workflows-scikit-learn-pipelines-part-3.html
# https://towardsdatascience.com/pipelines-custom-transformers-in-scikit-learn-the-step-by-step-guide-with-python-code-4a7d9b068156

# https://hub.packtpub.com/automl-build-machine-learning-pipeline-tutorial/

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

from sklearn import model_selection
from sklearn.model_selection import train_test_split, GridSearchCV

import joblib
from joblib import dump, load
import pickle

from sklearn.decomposition import PCA

from sklearn.inspection import permutation_importance

from sklearn.pipeline import Pipeline

# Algorithms
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

n_jobs = -1
trial  = 1

# Import database
with open('../../../Data/TCs_air5.txt') as f:
    lines = (line for line in f if not line.startswith('#'))
    dataset = np.loadtxt(lines, skiprows=1)

#dataset = np.loadtxt("../../../Data/TCs_air5.txt")
x = dataset[:,0:7] # T, P, x_N2, x_O2, x_NO, x_N, x_O
y = dataset[:,7:8]

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

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:',   y_train.shape)
print('Testing Features Shape:',  x_test.shape)
print('Testing Labels Shape:',    y_test.shape)

# prepare configuration for cross validation test harness
seed = 69

# Construct some pipelines
pipe_DT = Pipeline([('scl', StandardScaler()),
                    ('est', DecisionTreeRegressor(random_state=seed))])

#pipe_DT_PCA = Pipeline([('scl', StandardScaler()),
#			('pca', PCA(n_components=2)),
#                        ('est', DecisionTreeRegressor(random_state=seed))])

pipe_ET = Pipeline([('scl', StandardScaler()),
                    ('est', ExtraTreesRegressor(random_state=seed))])

#pipe_ET_PCA = Pipeline([('scl', StandardScaler()),
#			('pca', PCA(n_components=2)),
#                        ('est', ExtraTreesRegressor(random_state=seed))])

pipe_RF = Pipeline([('scl', StandardScaler()),
                    ('est', RandomForestRegressor(random_state=seed))])

#pipe_RF_PCA = Pipeline([('scl', StandardScaler()),
#			('pca', PCA(n_components=2)),
#			('est', RandomForestRegressor(random_state=seed))])

pipe_SVR = Pipeline([('scl', StandardScaler()),
                     ('est', SVR())])

#pipe_SVR_PCA = Pipeline([('scl', StandardScaler()),
#                         ('pca', PCA(n_components=2)),
#                         ('est', SVR())])

pipe_MLP = Pipeline([('scl', StandardScaler()),
                     ('est', MLPRegressor(random_state=seed))])

#pipe_MLP_PCA = Pipeline([('scl', StandardScaler()),
#                         ('pca', PCA(n_components=2)),
#                         ('est', MLPRegressor(random_state=seed))])

pipe_KN = Pipeline([('scl', StandardScaler()),
                    ('est', KNeighborsRegressor())])

#pipe_KN_PCA = Pipeline([('scl', StandardScaler()),
#                        ('pca', PCA(n_components=2)),
#                        ('est', KNeighborsRegressor())])

pipe_GB = Pipeline([('scl', StandardScaler()),
                    ('est', GradientBoostingRegressor(random_state=seed))])

#pipe_GB_PCA = Pipeline([('scl', StandardScaler()),
#                        ('pca', PCA(n_components=2)),
#                        ('est', GradientBoostingRegressor(random_state=seed))])

pipe_HGB = Pipeline([('scl', StandardScaler()),
                     ('est', HistGradientBoostingRegressor(random_state=seed))])

#pipe_HGB_PCA = Pipeline([('scl', StandardScaler()),
#                         ('pca', PCA(n_components=2)),
#                         ('est', HistGradientBoostingRegressor(random_state=seed))])

pipe_B = Pipeline([('scl', StandardScaler()),
                   ('est', BaggingRegressor(random_state=seed))])

#pipe_B_PCA = Pipeline([('scl', StandardScaler()),
#                       ('pca', PCA(n_components=2)),
#                       ('est', BaggingRegressor(random_state=seed))])

# Set grid search params
grid_params_DT = [{'est__criterion': ('mse', 'friedman_mse', 'mae'),
                   'est__splitter': ('best', 'random'),
                   'est__max_features': ('auto', 'sqrt', 'log2'),
}]

grid_params_RF = [{'est__n_estimators': (10, 100, 1000),
                   'est__min_weight_fraction_leaf': (0.0, 0.25, 0.5),
                   'est__max_features': ('sqrt','log2',None),
}]

grid_params_SVR = [{'est__kernel': ('poly', 'rbf',),
                    'est__gamma': ('scale', 'auto',),
                    'est__C': (1e0, 1e1, 1e2, 1e3,),
                    'est__epsilon': (1e-2, 1e-1,),
                    'est__coef0': (0.0, 0.1, 0.2, 0.3, 0.4, 0.5,),
}]

grid_params_ET = [{'est__n_estimators': (10, 100, 1000,),
                   'est__min_weight_fraction_leaf': (0.0, 0.25, 0.5,),
                   'est__max_features': ('sqrt','log2','auto', None,),
                   'est__max_samples': (1, 10, 100, 1000,),
                   'est__bootstrap': (True, False,),
                   'est__oob_score': (True, False,),
                   'est__warm_start': (True, False,),
                   'est__criterion': ('mse', 'mae',),
                   'est__max_depth': (1, 10, 100, None,),
                   'est__min_samples_split': (0.1, 0.25, 0.5, 0.75, 1.0,),
                   'est__min_samples_leaf': (1, 10, 100,),
}]

grid_params_KN = [{'est__algorithm': ('ball_tree', 'kd_tree', 'brute',),
                   'est__n_neighbors': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10,),
                   'est__leaf_size': (1, 10, 20, 30, 100,),
                   'est__weights': ('uniform', 'distance',),
                   'est__p': (1, 2,),
}]

grid_params_MLP = [{'est__activation' : ('tanh', 'relu',),
                    'est__solver' : ('lbfgs','adam','sgd',),
                    'est__learning_rate' : ('constant', 'invscaling', 'adaptive',),
                    'est__nesterovs_momentum': (True, False,),
}]

grid_params_GB = [{'est__n_estimators': (10, 100, 1000,),
                   'est__min_weight_fraction_leaf': (0.0, 0.25, 0.5,),
#                  'est__max_features': ('sqrt','log2','auto', None,),
#                  'est__max_features': (1,10,100,),
#                  'est__warm_start': (True, False,),
#                  'est__criterion': ('friedman_mse', 'mse', 'mae',),
#                  'est__max_depth': (1,10,100,None,),
                   'est__min_samples_split': (0.1,0.25,0.5,0.75,1.0,),
#                  'est__min_samples_leaf': (1,10,100,),
                   'est__loss': ('ls', 'lad', 'huber', 'quantile',),
}]

grid_params_HGB = [{
#                 'est__n_estimators': (10, 100, 1000,),
#                 'est__learning_rate': (0.1,0.01,0.001,),
#                 'est__max_leaf_nodes': (None,),
#                 'est__l2_regularization': (0., 0.25, 0.5, 0.75, 1.0,),
#                 'est__monotonic_cst': ([1, -1, 0]),
#                 'est__min_weight_fraction_leaf': (0.0, 0.25, 0.5,),
#                 'est__max_features': ('sqrt','log2','auto', None,),
#                 'est__warm_start': (True, False,),
#                 'est__criterion': ('friedman_mse', 'mse', 'mae',),
#                 'est__max_depth': (1,10,100,None,),
#                 'est__min_samples_split': (0.1,0.25,0.5,0.75,1.0,),
                  'est__min_samples_leaf': (1,10,100,),
                  'est__loss': ('least_squares', 'least_absolute_deviation', 'poisson',),
}]

grid_params_B = [{'est__n_estimators': (10, 100, 1000,),
                  'est__base_estimator': (SVR(),None,),
                  'est__max_samples': (1,10,100,1000,),
                  'est__max_features': (1,10,100,),
                  'est__bootstrap': (True, False,),
                  'est__bootstrap_features': (True, False,),
                  'est__oob_score': (True, False,),
                  'est__warm_start': (True, False,),
}]

# Construct grid searches
jobs = -1
scoring = 'r2'
verbose = 2
cv = 10

gs_DT = GridSearchCV(estimator=pipe_DT,
                     param_grid=grid_params_DT,
                     scoring=scoring,
                     verbose=verbose,
                     n_jobs=jobs,
                     cv=cv)

#gs_DT_PCA = GridSearchCV(estimator=pipe_DT_PCA,
#                         param_grid=grid_params_DT,
#                         scoring=scoring,
#                         verbose=verbose,
#                         n_jobs=jobs,
#                         cv=cv)

gs_ET = GridSearchCV(estimator=pipe_ET,
                     param_grid=grid_params_ET,
                     scoring=scoring,
                     verbose=verbose,
                     n_jobs=jobs,
                     cv=cv)

#gs_ET_PCA = GridSearchCV(estimator=pipe_ET_PCA,
#                         param_grid=grid_params_ET,
#                         scoring=scoring,
#                         verbose=verbose,
#                         n_jobs=jobs,
#                         cv=cv)

gs_RF = GridSearchCV(estimator=pipe_RF,
                     param_grid=grid_params_RF,
                     scoring=scoring,
                     verbose=verbose,
                     n_jobs=jobs,
                     cv=cv)

#gs_RF_PCA = GridSearchCV(estimator=pipe_RF_PCA,
#                         param_grid=grid_params_RF,
#                         scoring=scoring,
#                         verbose=verbose,
#                         n_jobs=jobs,
#                         cv=cv)

gs_GB = GridSearchCV(estimator=pipe_GB,
                     param_grid=grid_params_GB,
                     scoring=scoring,
                     verbose=verbose,
                     n_jobs=jobs,
                     cv=cv)

#gs_GB_PCA = GridSearchCV(estimator=pipe_GB_PCA,
#                         param_grid=grid_params_GB,
#                         scoring=scoring,
#                         verbose=verbose,
#                         n_jobs=jobs,
#                         cv=cv)

gs_HGB = GridSearchCV(estimator=pipe_HGB,
                      param_grid=grid_params_HGB,
                      scoring=scoring,
                      verbose=verbose,
                      n_jobs=jobs,
                      cv=cv)

#gs_HGB_PCA = GridSearchCV(estimator=pipe_HGB_PCA,
#                          param_grid=grid_params_HGB,
#                          scoring=scoring,
#                          verbose=verbose,
#                          n_jobs=jobs,
#                          cv=cv)

gs_B = GridSearchCV(estimator=pipe_B,
                    param_grid=grid_params_B,
                    scoring=scoring,
                    verbose=verbose,
                    n_jobs=jobs,
                    cv=cv)

#gs_B_PCA = GridSearchCV(estimator=pipe_B_PCA,
#                        param_grid=grid_params_B,
#                        scoring=scoring,
#                        verbose=verbose,
#                        n_jobs=jobs,
#                        cv=cv)

gs_MLP = GridSearchCV(estimator=pipe_MLP,
                      param_grid=grid_params_MLP,
                      scoring=scoring,
                      verbose=verbose,
                      n_jobs=jobs,
                      cv=cv)

#gs_MLP_PCA = GridSearchCV(estimator=pipe_MLP_PCA,
#                          param_grid=grid_params_MLP,
#                          scoring=scoring,
#                          verbose=verbose,
#                          n_jobs=jobs,
#                          cv=cv)

gs_SVR = GridSearchCV(estimator=pipe_SVR,
                      param_grid=grid_params_SVR,
                      scoring=scoring,
                      verbose=verbose,
                      n_jobs=jobs,
                      cv=cv)

#gs_SVR_PCA = GridSearchCV(estimator=pipe_SVR_PCA,
#                          param_grid=grid_params_SVR,
#                          scoring=scoring,
#                          verbose=verbose,
#                          n_jobs=jobs,
#                          cv=cv)

gs_KN = GridSearchCV(estimator=pipe_KN,
                     param_grid=grid_params_KN,
                     scoring=scoring,
                     verbose=verbose,
                     n_jobs=jobs,
                     cv=cv)

#gs_KN_PCA = GridSearchCV(estimator=pipe_KN_PCA,
#                         param_grid=grid_params_KN,
#                         scoring=scoring,
#                         verbose=verbose,
#                         n_jobs=jobs,
#                         cv=cv)

# List of pipelines for ease of iteration
grids = [gs_ET,  #gs_DT_PCA,
         gs_DT,  #gs_ET_PCA,
         gs_RF,  #gs_RF_PCA,
         gs_SVR, #gs_SVR_PCA,
         gs_GB,  #gs_GB_PCA,
         gs_HGB, #gs_HGB_PCA,
         gs_B,   #gs_B_PCA,
         gs_KN,  #gs_KN_PCA,
         gs_MLP, #gs_MLP_PCA,
]

# Dictionary of pipelines and estimator types for ease of reference
#grid_dict = {0:  'DT', 1:  'DT w/PCA',
#             2:  'ET', 3:  'ET w/PCA',
#             4:  'RF', 5:  'RF w/PCA',
#             6:  'SVR',7:  'SVR w/PCA',
#             8:  'GB', 9:  'GB w/PCA',
#             10: 'HGB',11: 'HGB w/PCA',
#             12: 'B',  13: 'B w/PCA',
#             14: 'KN', 15: 'KN w/PCA',
#             16: 'MLP',17: 'MLP w/PCA'
#}

grid_dict = {0: 'ET',
             1: 'DT',
             2: 'RF',
             3: 'SVR',
             4: 'GB',
             5: 'HGB',
             6: 'B',
             7: 'KN',
             8: 'MLP',
}

# Fit the grid search objects
print('Performing model optimizations ...')
best_r2 = 0.0
best_est = 0
best_gs = ''
for idx, gs in enumerate(grids):

	print('\nEstimator: %s' % grid_dict[idx])

	# Fit grid search
	gs.fit(x_train, y_train.ravel())

	# Best params
	print('Best params: %s' % gs.best_params_)

	# Best training data score
	print('Best training score: %.6f' % gs.best_score_)

	# Train data score of model with best params
	print('Train set R2  score for best params: %.6f ' % r2_score(                y_train, gs.predict(x_train))
	print('Train set ME  score for best params: %.6f ' % max_error(               y_train, gs.predict(x_train))
	print('Train set MSE score for best params: %.6f ' % mean_squared_error(      y_train, gs.predict(x_train))
	print('Train set MAE score for best params: %.6f ' % mean_absolute_error(     y_train, gs.predict(x_train))
	print('Train set MSE score for best params: %.6f ' % mean_squared_error(      y_train, gs.predict(x_train))
	print('Train set EVS score for best params: %.6f ' % explained_variance_score(y_train, gs.predict(x_train))

	# Predict on test data with best params
	y_pred = gs.predict(x_test)

	# Test data score of model with best params
	print('Test set R2  score for best params: %.6f ' % r2_score(                y_test, y_pred))
	print('Test set ME  score for best params: %.6f ' % max_error(               y_test, y_pred))
	print('Test set MSE score for best params: %.6f ' % mean_squared_error(      y_test, y_pred))
	print('Test set MAE score for best params: %.6f ' % mean_absolute_error(     y_test, y_pred))
	print('Test set MSE score for best params: %.6f ' % mean_squared_error(      y_test, y_pred))
	print('Test set EVS score for best params: %.6f ' % explained_variance_score(y_test, y_pred))

	# Track best (highest test score) model
	if r2_score(y_test, y_pred) > best_r2:
		best_r2 = r2_score(y_test, y_pred)
		best_gs = gs
		best_est = idx

print('\nEstimator with best test set score: %s' % grid_dict[best_est])

# Save best grid search pipeline to file
dump_file = 'best_gs_pipeline.pkl'
joblib.dump(best_gs, dump_file, compress=1)
print('\nSaved %s grid search pipeline to file: %s' % (grid_dict[best_est], dump_file))
