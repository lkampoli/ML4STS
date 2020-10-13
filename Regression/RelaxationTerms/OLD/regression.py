#!/usr/bin/env python

import time
import sys
sys.path.insert(0, '../Utilities/')
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
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, cross_val_score
from sklearn import kernel_ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn import neighbors
from sklearn.neighbors import NearestNeighbors
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import svm
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

n_jobs = 1
trial  = 1

dataset=np.loadtxt("./data/datarelax.txt")

# ... only for plotting
#dataset=np.loadtxt("../data/datarelax.txt")
#x=dataset[:,0:1]   # Temperatures
#y=dataset[:,1:50]  # Rci (relaxation source terms)

#for i in range (2,48):
#    plt.scatter(dataset[:,0:1], dataset[:,i], s=0.5, label=i)

#plt.title('$R_{ci}$ for $N_2/N$')
#plt.xlabel('T [K]')
#plt.ylabel('$R_{ci}$ $[J/m^3/s]$')
##plt.legend()
#plt.tight_layout()
#plt.savefig("relaxation_source_terms.pdf")
#plt.show()

# Here, I learn one specific level of R_ci spanning all temperatures
x=dataset[:,0:1]   # Temperatures
y=dataset[:,9:10]  # Rci (relaxation source terms)

# Here, I fix the temperature and learn all levels of R_ci
#x=dataset[150,0:1]   # Temperatures
#y=dataset[150,1:50]  # Rci (relaxation source terms)

# TODO: Here, I want to learn all T and all Rci alltogether
#x=dataset[:,0:1]   # Temperatures
#y=dataset[:,1:50]  # Rci (relaxation source terms)

# 2D Plot
#plt.scatter(x, y, s=0.5)
#plt.title('$R_{ci}$ for $N_2/N$ and i = 10')
#plt.xlabel('T [K]')
#plt.ylabel('$R_{ci}$ $[J/m^3/s]$')
#plt.tight_layout()
#plt.savefig("relaxation_source_terms.pdf")
#plt.show()

y=np.reshape(y, (-1,1))
sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(x)
Y = sc_y.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.75, test_size=0.25, random_state=42)

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', x_test.shape)
print('Testing Labels Shape:', y_test.shape)

# KernelRidge
#hyper_params = [{'kernel': ('poly','rbf',), 'alpha': (1e-4,1e-2,0.1,1,10,), 'gamma': (0.01,0.1,1,10,100,),}]

# k-kearest neighbor
#hyper_params = [{'algorithm': ('ball_tree', 'kd_tree', 'brute',), 'n_neighbors': (1,2,3,4,5,6,7,8,9,10,),
#                 'leaf_size': (1, 10, 20, 30, 100,), 'weights': ('uniform', 'distance',), 'p': (1,2,),}]

# Random Forest
#hyper_params = [{'n_estimators': (10, 100, 1000),
#                 'min_weight_fraction_leaf': (0.0, 0.25, 0.5),
#                 'max_features': ('sqrt','log2',None),
#}]

# Extra Trees
#hyper_params = [{'n_estimators': (10, 100, 1000,),
#                 'min_weight_fraction_leaf': (0.0, 0.25, 0.5,),
#                 'max_features': ('sqrt','log2','auto', None,),
#                 'max_samples': (1,10,100,1000,),
#                 'bootstrap': (True, False,),
#                 'oob_score': (True, False,),
#                 'warm_start': (True, False,),
#                 'criterion': ('mse', 'mae',),
#                 'max_depth': (1,10,100,None,),
#                 'max_leaf_nodes': (1,10,100,),
#                 'min_samples_split': (0.1,0.25,0.5,0.75,1.0,),
#                 'min_samples_leaf': (1,10,100,),
#}]

# Support Vector Machines
#hyper_params = [{'kernel': ('poly', 'rbf',), 'gamma': ('scale', 'auto',),
#                 'C': (1e-2, 1e-1, 1e0, 1e1, 1e2,), 'epsilon': (1e-2, 1e-1, 1e0, 1e1, 1e2,), }]

## GradientBoosting
#hyper_params = [{'n_estimators': (10, 100, 1000,),
#                 'min_weight_fraction_leaf': (0.0, 0.25, 0.5,),
##                 'max_features': ('sqrt','log2','auto', None,),
##                 'warm_start': (True, False,),
##                 'criterion': ('friedman_mse', 'mse', 'mae',),
##                 'max_depth': (1,10,100,None,),
#                 'min_samples_split': (2,5,10,100,), #0.1,0.25,0.5,0.75,1.0,),
#                 'min_samples_leaf': (2,5,10,100,),
#                 'loss': ('ls', 'lad', 'huber', 'quantile',),
#                 # 'subsample':
#                 # 'learning_rate':
#}]

#est=ensemble.RandomForestRegressor()
#est=kernel_ridge.KernelRidge()
#est=neighbors.NearestNeighbors()
#est=neighbors.KNeighborsRegressor()
#est=ensemble.ExtraTreesRegressor()
#est=svm.SVR()
#est=ensemble.GradientBoostingRegressor()

#gs = GridSearchCV(est, cv=5, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2')

#t0 = time.time()
#gs.fit(x_train, y_train.ravel())
#runtime = time.time() - t0
#print("Complexity and bandwidth selected and model fitted in %.6f s" % runtime)

#train_score_mse = mean_squared_error(      sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
#train_score_mae = mean_absolute_error(     sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
#train_score_evs = explained_variance_score(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
#train_score_me  = max_error(               sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))

#test_score_mse  = mean_squared_error(      sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))
#test_score_mae  = mean_absolute_error(     sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))
#test_score_evs  = explained_variance_score(sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))
#test_score_me   = max_error(               sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))

#sorted_grid_params = sorted(gs.best_params_.items(), key=operator.itemgetter(0))

#out_text = '\t'.join(['regression',
#                      str(trial),
#                      str(sorted_grid_params).replace('\n',','),
#                      str(train_score_mse),
#                      str(train_score_mae),
#                      str(train_score_evs),
#                      str(train_score_me),
#                      str(test_score_mse),
#                      str(test_score_mae),
#                      str(test_score_evs),
#                      str(test_score_me),
#                      str(runtime)])
#print(out_text)
#sys.stdout.flush()

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
#best_kernel = gs.best_params_['kernel']
#best_gamma = gs.best_params_['gamma']
#best_C = gs.best_params_['C']
#best_epsilon = gs.best_params_['epsilon']

## GB
#best_n_estimators = gs.best_params_['n_estimators']
#best_min_weight_fraction_leaf = gs.best_params_['min_weight_fraction_leaf']
##best_max_features = gs.best_params_['max_features']
##best_warm_start = gs.best_params_['warm_start']
##best_criterion = gs.best_params_['criterion']
##best_max_depth = gs.best_params_['max_depth']
#best_min_samples_split = gs.best_params_['min_samples_split']
#best_loss = gs.best_params_['loss']
#best_min_samples_leaf = gs.best_params_['min_samples_leaf']

#outF = open("output.txt", "w")
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
#print('best_kernel = ', best_kernel, file=outF)
#print('best_gamma = ', best_gamma, file=outF)
#print('best_C = ', best_C, file=outF)
#print('best_epsilon = ', best_epsilon, file=outF)
#
#print('best_n_estimators = ', best_n_estimators, file=outF)
#print('best_min_weight_fraction_leaf = ', best_min_weight_fraction_leaf, file=outF)
##print('best_max_features = ', best_max_features, file=outF)
##print('best_warm_start = ', best_warm_start, file=outF)
##print('best_criterion = ', best_criterion, file=outF)
##print('best_max_depth = ', best_max_depth, file=outF)
#print('best_min_samples_split = ', best_min_samples_split, file=outF)
#print('best_min_samples_leaf = ', best_min_samples_leaf, file=outF)
#print('best_loss = ', best_loss, file=outF)
#outF.close()

# KN
KN = KNeighborsRegressor(n_neighbors=6,
                         algorithm='ball_tree',
                         leaf_size=1,
                         weights='distance',
                         p=1)

t0 = time.time()
KN.fit(x_train, y_train.ravel())
KN_fit = time.time() - t0
print("KN complexity and bandwidth selected and model fitted in %.6f s" % KN_fit)

t0 = time.time()
y_KN = KN.predict(x_test)
KN_predict = time.time() - t0
print("KN prediction for %d inputs in %.6f s" % (x_test.shape[0], KN_predict))

print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_KN))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_KN))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_KN)))

# KR
KR = KernelRidge(kernel='rbf',
                 gamma=10,
                 alpha=0.0001)

t0 = time.time()
KR.fit(x_train, y_train.ravel())
KR_fit = time.time() - t0
print("KR complexity and bandwidth selected and model fitted in %.6f s" % KR_fit)

t0 = time.time()
y_KR = KR.predict(x_test)
KR_predict = time.time() - t0
print("KR prediction for %d inputs in %.6f s" % (x_test.shape[0], KR_predict))

print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_KR))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_KR))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_KR)))

# RF
RF = RandomForestRegressor(n_estimators=100,
                           min_weight_fraction_leaf=0.0,
                           random_state=69)

t0 = time.time()
RF.fit(x_train, y_train.ravel())
RF_fit = time.time() - t0
print("RF complexity and bandwidth selected and model fitted in %.6f s" % RF_fit)

t0 = time.time()
y_RF = RF.predict(x_test)
RF_predict = time.time() - t0
print("RF prediction for %d inputs in %.6f s" % (x_test.shape[0], RF_predict))

print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_RF))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_RF))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_RF)))

# ET
ET = ExtraTreesRegressor(n_estimators=1000,
                         min_weight_fraction_leaf=0.0,
#                        max_features=best_max_features,
#                        bootstrap=best_bootstrap,
#                        oob_score=best_oob_score,
#                        warm_start=best_warm_start,
#                        criterion=best_criterion,
#                        max_depth=best_max_depth,
                         max_leaf_nodes=100,
                         min_samples_split=2,
                         min_samples_leaf=1,
                         random_state=69)

t0 = time.time()
ET.fit(x_train, y_train.ravel())
ET_fit = time.time() - t0
print("ET complexity and bandwidth selected and model fitted in %.6f s" % ET_fit)

t0 = time.time()
y_ET = ET.predict(x_test)
ET_predict = time.time() - t0
print("ET prediction for %d inputs in %.6f s" % (x_test.shape[0], ET_predict))

print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_ET))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_ET))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_ET)))

# SVR
SVR = SVR(kernel='rbf',
          epsilon=0.01,
          C=100.,
          gamma='scale')

t0 = time.time()
SVR.fit(x_train, y_train.ravel())
SVR_fit = time.time() - t0
print("SVR complexity and bandwidth selected and model fitted in %.6f s" % SVR_fit)

t0 = time.time()
y_SVR = SVR.predict(x_test)
SVR_predict = time.time() - t0
print("SVR prediction for %d inputs in %.6f s" % (x_test.shape[0], SVR_predict))

print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_SVR))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_SVR))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_SVR)))

GB = GradientBoostingRegressor(n_estimators=100,
                               min_weight_fraction_leaf=0.0,
#                              max_features=best_max_features,
#                              warm_start=best_warm_start,
#                              criterion=best_criterion,
#                              max_depth=best_max_depth,
                               loss='ls',
                               min_samples_split=5,
                               min_samples_leaf=2,
                               random_state=69)

t0 = time.time()
GB.fit(x_train, y_train.ravel())
GB_fit = time.time() - t0
print("Complexity and bandwidth selected and model fitted in %.6f s" % GB_fit)

t0 = time.time()
y_GB = GB.predict(x_test)
GB_predict = time.time() - t0
print("Prediction for %d inputs in %.6f s" % (x_test.shape[0], GB_predict))

print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_GB))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_GB))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_GB)))

# MLP
MLP = MLPRegressor(hidden_layer_sizes=(35,35),
                   activation='tanh',
                   solver='lbfgs',
                   learning_rate='constant',
                   nesterovs_momentum=False,
                   max_iter=1000,
                   random_state=69)

t0 = time.time()
MLP.fit(x_train, y_train.ravel())
MLP_fit = time.time() - t0
print("MLP complexity and bandwidth selected and model fitted in %.6f s" % MLP_fit)

t0 = time.time()
y_MLP = MLP.predict(x_test)
MLP_predict = time.time() - t0
print("MLP prediction for %d inputs in %.6f s" % (x_test.shape[0], MLP_predict))

print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_MLP))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_MLP))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_MLP)))

# open a file to append
#outF = open("output.txt", "a")
#print("Complexity and bandwidth selected and model fitted in %.6f s" % regr_fit, file=outF)
#print("Prediction for %d inputs in %.6f s" % (x_test.shape[0], regr_predict),file=outF)
#print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_regr), file=outF)
#print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_regr), file=outF)
#print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_regr)), file=outF)
#outF.close()

x_test_dim = sc_x.inverse_transform(x_test)
y_test_dim = sc_y.inverse_transform(y_test)
y_KN_dim   = sc_y.inverse_transform(y_KN)
y_KR_dim   = sc_y.inverse_transform(y_KR)
y_RF_dim   = sc_y.inverse_transform(y_RF)
y_SVR_dim  = sc_y.inverse_transform(y_SVR)
y_GB_dim   = sc_y.inverse_transform(y_GB)
y_ET_dim   = sc_y.inverse_transform(y_ET)
y_MLP_dim  = sc_y.inverse_transform(y_MLP)

plt.scatter(x_test_dim, y_test_dim, s=6, c='k', marker='o', label='KAPPA')
plt.scatter(x_test_dim, y_KN_dim,   s=6, c='r', marker='+', label='KN')
plt.scatter(x_test_dim, y_KR_dim,   s=6, c='b', marker='x', label='KR')
plt.scatter(x_test_dim, y_RF_dim,   s=6, c='g', marker='v', label='RF')
plt.scatter(x_test_dim, y_SVR_dim,  s=6, c='y', marker='^', label='SVR')
plt.scatter(x_test_dim, y_GB_dim,   s=6, c='m', marker='<', label='GB')
plt.scatter(x_test_dim, y_ET_dim,   s=6, c='c', marker='>', label='ET')
plt.scatter(x_test_dim, y_MLP_dim,  s=6, c='purple', marker='H', label='MLP')
plt.title('Relaxation term $R_{ci}$ regression')
plt.ylabel('$R_{ci}$ $[J/m^3/s]$')
plt.xlabel('T [K] ')
plt.legend()
plt.tight_layout()
plt.savefig("regression.eps", dpi=150, crop='false')
plt.show()
