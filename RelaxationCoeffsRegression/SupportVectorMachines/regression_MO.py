#!/usr/bin/env python

import time
import sys
sys.path.insert(0, '../../Utilities/')
from plotting import newfig, savefig
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import operator
import itertools
from sklearn import metrics
from sklearn.metrics import *
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV #, learning_curve, cross_val_score
from sklearn import svm
from sklearn.svm import SVR
from joblib import dump, load
import pickle
from sklearn.multioutput import MultiOutputRegressor
from sklearn.multioutput import RegressorChain
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from numpy import mean
from numpy import std
from numpy import absolute

n_jobs = 1
trial  = 1

dataset=np.loadtxt("../data/datarelax.txt")

x=dataset[:,0:1]  # Temperatures
y=dataset[:,1:50] # Rci (relaxation source terms)

#x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.75, test_size=0.25, random_state=42)
#x_train_sc, x_test_sc, y_train_sc, y_test_sc = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=42)
x_train_sc, x_test_sc, y_train_sc, y_test_sc = train_test_split(x, y, train_size=0.80, test_size=0.20, random_state=42)

sc_x = StandardScaler()
sc_y = StandardScaler()

x_train = sc_x.fit_transform(x_train_sc)
y_train = sc_y.fit_transform(y_train_sc)
x_test  = sc_x.fit_transform(x_test_sc)
y_test  = sc_y.fit_transform(y_test_sc)

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', x_test.shape)
print('Testing Labels Shape:', y_test.shape)

# Support Vector Machines
hyper_params = [{
                 'kernel': ('poly', 'rbf',),
                 #'kernel': ('rbf',),
                 #'gamma': ('scale',),
                 'gamma': ('scale', 'auto',),
                 'C': (1.0, 2.,3.,4.,5.,10.,100., 1000.,),
                 #'C': (500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 10000),
                 #'C': (1, 10, 50, 75, 100, 200, 500, 750, 1000,),
                 #'C': (1e-2, 1e-1, 1e0, 1e1, 1e2,),
                 #'epsilon': (0.01, 0.025, 0.05, 0.075, 0.1,),
                 'epsilon': (0.01, 0.015, 0.020, 0.025, 0.030, 0.035,),
                 #'epsilon': (1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4),
                 #'coef0': (0.0, 0.1, 0.2, 0.3, 0.4, 0.5,),
                 'coef0': (0.0,),
                 #'coef0': (0.0, 0.25, 0.5, 0.75, 1.0,),
                 }]

est = svm.SVR()
regr = MultiOutputRegressor(estimator=est)
#regr = RegressorChain(base_estimator=est)

#gs = GridSearchCV(est, cv=10, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2')

# define the evaluation procedure
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# evaluate the model and collect the scores
n_scores = cross_val_score(regr, x_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
#scores = cross_val_score(regr, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)

# force the scores to be positive
n_scores = absolute(n_scores)

# summarize performance
print('MAE: %.6f (%.6f)' % (mean(n_scores), std(n_scores)))
#print('Accuracy: %.6f (%.6f)' % (mean(scores), std(scores)))

t0 = time.time()
gs.fit(x_train, y_train)
runtime = time.time() - t0
print("Complexity and bandwidth selected and model fitted in %.6f s" % runtime)

# save the model to disk
dump(gs, 'model_MO_SVR.sav')

#train_score_mse = mean_squared_error(      sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
#train_score_mae = mean_absolute_error(     sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
#train_score_evs = explained_variance_score(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
#train_score_me  = max_error(               sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
#
#test_score_mse  = mean_squared_error(      sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))
#test_score_mae  = mean_absolute_error(     sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))
#test_score_evs  = explained_variance_score(sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))
#test_score_me   = max_error(               sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))
#test_score_r2   = r2_score(                sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))

#print("The model performance for testing set")
#print("--------------------------------------")
#print('MAE is {}'.format(test_score_mae))
#print('MSE is {}'.format(test_score_mse))
#print('EVS is {}'.format(test_score_evs))
#print('ME is {}'.format(test_score_me))
#print('R2 score is {}'.format(test_score_r2))

#feature_importances = gs.best_estimator_.feature_importances_
#print(feature_importances)

sorted_grid_params = sorted(gs.best_params_.items(), key=operator.itemgetter(0))

print(gs.cv_results_)
print(gs.best_params_)

out_text = '\t'.join(['regression',
                      str(trial),
                      str(sorted_grid_params).replace('\n',','),
                      str(train_score_mse),
                      str(train_score_mae),
                      str(train_score_evs),
                      str(train_score_me),
                      str(test_score_mse),
                      str(test_score_mae),
                      str(test_score_evs),
                      str(test_score_me),
                      str(runtime)])
print(out_text)
sys.stdout.flush()

# KernelRidge
#best_algorithm   = gs.best_params_['algorithm']
#best_n_neighbors = gs.best_params_['n_neighbors']
#best_leaf_size   = gs.best_params_['leaf_size']
#best_weights     = gs.best_params_['weights']
#best_p           = gs.best_params_['p']

# kNearestNeighbour
#best_kernel       = gs.best_params_['kernel']
#best_alpha        = gs.best_params_['alpha']
#best_gamma        = gs.best_params_['gamma']

# RandomForest
#best_n_estimators = gs.best_params_['n_estimators']
#best_min_weight_fraction_leaf = gs.best_params_['min_weight_fraction_leaf']
#best_max_features = gs.best_params_['max_features']

# ExtraTrees
#best_n_estimators = gs.best_params_['n_estimators']
#best_min_weight_fraction_leaf = gs.best_params_['min_weight_fraction_leaf']
#best_max_features = gs.best_params_['max_features']
#best_max_samples = gs.best_params_['max_samples']
#best_bootstrap = gs.best_params_['bootstrap']
#best_oob_score = gs.best_params_['oob_score']
#best_warm_start = gs.best_params_['warm_start']
#best_criterion = gs.best_params_['criterion']
#best_max_depth = gs.best_params_['max_depth']
#best_min_samples_split = gs.best_params_['min_samples_split']
#best_min_samples_leaf = gs.best_params_['min_samples_leaf']
#best_max_leaf_nodes = gs.best_params_['max_leaf_nodes']

# SVR
best_kernel = gs.best_params_['kernel']
best_gamma = gs.best_params_['gamma']
best_C = gs.best_params_['C']
best_epsilon = gs.best_params_['epsilon']
best_coef0 = gs.best_params_['coef0']

outF = open("output.txt", "w")
#print('best_algorithm = ', best_algorithm, file=outF)
#print('best_n_neighbors = ', best_n_neighbors, file=outF)
#print('best_leaf_size = ', best_leaf_size, file=outF)
#print('best_weights = ', best_weights, file=outF)
#print('best_p = ', best_p, file=outF)
#
#print('best_kernel = ', best_kernel, file=outF)
#print('best_alpha = ', best_alpha, file=outF)
#print('best_gamma = ', best_gamma, file=outF)
#
#print('best_n_estimators = ', best_n_estimators, file=outF)
#print('best_min_weight_fraction_leaf = ', best_min_weight_fraction_leaf, file=outF)
#print('best_max_features = ', best_max_features, file=outF)
#
#print('best_n_estimators = ', best_n_estimators, file=outF)
#print('best_min_weight_fraction_leaf = ', best_min_weight_fraction_leaf, file=outF)
#print('best_max_features = ', best_max_features, file=outF)
#print('best_bootstrap = ', best_bootstrap, file=outF)
#print('best_oob_score = ', best_oob_score, file=outF)
#print('best_warm_start = ', best_warm_start, file=outF)
#print('best_criterion = ', best_criterion, file=outF)
#print('best_max_depth = ', best_max_depth, file=outF)
#print('best_min_samples_split = ', best_min_samples_split, file=outF)
#print('best_min_samples_leaf = ', best_min_samples_leaf, file=outF)
#print('best_min_samples_leaf = ', best_min_samples_leaf, file=outF)
#print('best_max_leaf_nodes = ', best_max_leaf_nodes, file=outF)
#
print('best_kernel = ', best_kernel, file=outF)
print('best_gamma = ', best_gamma, file=outF)
print('best_C = ', best_C, file=outF)
print('best_epsilon = ', best_epsilon, file=outF)
print('best_coef0 = ', best_coef0, file=outF)
outF.close()

#regr = KNeighborsRegressor(n_neighbors=best_n_neighbors, algorithm=best_algorithm,
#                         leaf_size=best_leaf_size, weights=best_weights, p=best_p)
#
#regr = KernelRidge(kernel=best_kernel, gamma=best_gamma, alpha=best_alpha)
#
#regr = RandomForestRegressor(n_estimators=best_n_estimators,
#                             min_weight_fraction_leaf=best_min_weight_fraction_leaf,
#                             max_features=best_max_features)
#
#regr = ExtraTreesRegressor(n_estimators=best_n_estimators,
#                           min_weight_fraction_leaf=best_min_weight_fraction_leaf,
#                           max_features=best_max_features,
#                           bootstrap=best_bootstrap,
#                           oob_score=best_oob_score,
#                           warm_start=best_warm_start,
#                           criterion=best_criterion,
#                           max_depth=best_max_depth,
#                           max_leaf_nodes=best_max_leaf_nodes,
#                           min_samples_split=best_min_samples_split,
#                           min_samples_leaf=best_min_samples_leaf)
#
regr = SVR(kernel=best_kernel,
           epsilon=best_epsilon,
           C=best_C,
           gamma=best_gamma,
           coef0=best_coef0)

t0 = time.time()
regr.fit(x_train, y_train.ravel())
regr_fit = time.time() - t0
print("Complexity and bandwidth selected and model fitted in %.6f s" % regr_fit)

t0 = time.time()
y_regr = regr.predict(x_test)
regr_predict = time.time() - t0
print("Prediction for %d inputs in %.6f s" % (x_test.shape[0], regr_predict))

# open a file to append
outF = open("output.txt", "a")
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

plt.scatter(x_test_dim, y_test_dim, s=5, c='r', marker='o', label='Matlab')
plt.scatter(x_test_dim, y_regr_dim, s=2, c='k', marker='d', label='SupportVectorMachine')
#plt.title('Relaxation term $R_{ci}$ regression')
plt.ylabel('$R_{ci}$ $[J/m^3/s]$')
plt.xlabel('T [K] ')
plt.legend()
plt.tight_layout()
plt.savefig("regression_SVR.eps", dpi=150, crop='false')
plt.savefig("regression_SVR.pdf", dpi=150, crop='false')
plt.show()



#from sklearn.multioutput import MultiOutputRegressor
#from sklearn.svm import SVR
#from sklearn.model_selection import GridSearchCV
#from sklearn.pipeline import Pipeline
#
#pipe_svr = Pipeline([('scl', StandardScaler()),
#        ('reg', MultiOutputRegressor(SVR()))])
#
#grid_param_svr = {
#    'reg__estimator__C': [0.1,1,10]
#}
#
#gs_svr = (GridSearchCV(estimator=pipe_svr,
#                      param_grid=grid_param_svr,
#                      cv=2,
#                      scoring = 'neg_mean_squared_error',
#                      n_jobs = -1))
#
#gs_svr = gs_svr.fit(X_train,y_train)
#gs_svr.best_estimator_
#
#Pipeline(steps=[('scl', StandardScaler(copy=True, with_mean=True, with_std=True)),
#('reg', MultiOutputRegressor(estimator=SVR(C=10, cache_size=200,
# coef0=0.0, degree=3, epsilon=0.1, gamma='auto', kernel='rbf', max_iter=-1,
# shrinking=True, tol=0.001, verbose=False), n_jobs=1))])
