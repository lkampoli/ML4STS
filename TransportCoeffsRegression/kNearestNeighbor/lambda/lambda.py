#!/usr/bin/env python

import time

import sys
sys.path.insert(0, '../../../Utilities/')

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

from sklearn import neighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.neighbors import NearestNeighbors

n_jobs = 1
trial  = 1

# Import database
dataset=np.loadtxt("../../data/dataset_lite.csv", delimiter=",")
x=dataset[:,0:2]
y=dataset[:,4] # 0: X, 1: T, 2: shear, 3: bulk, 4: conductivity

# Plot dataset
#plt.scatter(x[:,1], dataset[:,4], s=0.5)
#plt.title('Themal conductivity')
#plt.xlabel('T [K]')
#plt.ylabel(r'$\lambda$')
#plt.show()

y=np.reshape(y, (-1,1))
sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(x)
Y = sc_y.fit_transform(y)

# The data is then split into training and test data
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.75, test_size=0.25, random_state=42)

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', x_test.shape)
print('Testing Labels Shape:', y_test.shape)

# kNN
hyper_params = [{'algorithm': ('ball_tree', 'kd_tree', 'brute',), 'n_neighbors': (1,2,3,4,5,6,7,8,9,10,),
                 'leaf_size': (1, 10, 20, 30, 100,), 'weights': ('uniform', 'distance',), 'p': (1,2,),}]

est=neighbors.KNeighborsRegressor()
grid_clf = GridSearchCV(est, cv=5, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2')

t0 = time.time()
grid_clf.fit(x_train, y_train.ravel())
runtime = time.time() - t0
print("kNN complexity and bandwidth selected and model fitted in %.3f s" % runtime)

train_score_mse  = mean_squared_error(sc_y.inverse_transform(y_train), sc_y.inverse_transform(grid_clf.predict(x_train)))
train_score_mae  = mean_absolute_error(sc_y.inverse_transform(y_train),sc_y.inverse_transform(grid_clf.predict(x_train)))
train_score_evs  = explained_variance_score(sc_y.inverse_transform(y_train), sc_y.inverse_transform(grid_clf.predict(x_train)))
train_score_me   = max_error(sc_y.inverse_transform(y_train), sc_y.inverse_transform(grid_clf.predict(x_train)))
#train_score_msle = mean_squared_log_error(sc_y.inverse_transform(y_train), sc_y.inverse_transform(grid_clf.predict(x_train)))

test_score_mse  = mean_squared_error(sc_y.inverse_transform(y_test),  sc_y.inverse_transform(grid_clf.predict(x_test)))
test_score_mae  = mean_absolute_error(sc_y.inverse_transform(y_test), sc_y.inverse_transform(grid_clf.predict(x_test)))
test_score_evs  = explained_variance_score(sc_y.inverse_transform(y_test),  sc_y.inverse_transform(grid_clf.predict(x_test)))
test_score_me   = max_error(sc_y.inverse_transform(y_test),  sc_y.inverse_transform(grid_clf.predict(x_test)))
#test_score_msle = mean_squared_log_error(sc_y.inverse_transform(y_test),  sc_y.inverse_transform(grid_clf.predict(x_test)))

sorted_grid_params = sorted(grid_clf.best_params_.items(), key=operator.itemgetter(0))

out_text = '\t'.join(['k-nearest-neighbour',
                      str(trial),
                      str(sorted_grid_params).replace('\n',','),
                      str(train_score_mse),
                      str(train_score_mae),
                      str(train_score_evs),
                      str(train_score_me),
#                      str(train_score_msle),
                      str(test_score_mse),
                      str(test_score_mae),
                      str(test_score_evs),
                      str(test_score_me),
#                      str(test_score_msle),
                      str(runtime)])
print(out_text)
sys.stdout.flush()

best_algorithm = grid_clf.best_params_['algorithm']
best_n_neighbors = grid_clf.best_params_['n_neighbors']
best_leaf_size = grid_clf.best_params_['leaf_size']
best_weights = grid_clf.best_params_['weights']
best_p = grid_clf.best_params_['p']

# open a (new) file to write
outF = open("output.txt", "w")
print('best_algorithm = ', best_algorithm, file=outF)
print('best_n_neighbors = ', best_n_neighbors, file=outF)
print('best_leaf_size = ', best_leaf_size, file=outF)
print('best_weights = ', best_weights, file=outF)
print('best_p = ', best_p, file=outF)
outF.close()

kn = KNeighborsRegressor(n_neighbors=best_n_neighbors, algorithm=best_algorithm,
                         leaf_size=best_leaf_size, weights=best_weights, p=best_p)

t0 = time.time()
kn.fit(x_train, y_train.ravel())
kn_fit = time.time() - t0
print("kNN complexity and bandwidth selected and model fitted in %.3f s" % kn_fit)

t0 = time.time()
y_kn = kn.predict(x_test)
kn_predict = time.time() - t0
print("kNN prediction for %d inputs in %.3f s" % (x_test.shape[0], kn_predict))

# open a file to append
outF = open("output.txt", "a")
print("kNN complexity and bandwidth selected and model fitted in %.3f s" % kn_fit, file=outF)
print("kNN prediction for %d inputs in %.3f s" % (x_test.shape[0], kn_predict),file=outF)
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_kn), file=outF)
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_kn), file=outF)
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_kn)), file=outF)
outF.close()

print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_kn))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_kn))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_kn)))

x_test_dim = sc_x.inverse_transform(x_test)
y_test_dim = sc_y.inverse_transform(y_test)
y_kn_dim   = sc_y.inverse_transform(y_kn)

plt.scatter(x_test_dim[:,1], y_test_dim[:], s=5, c='red',     marker='o', label='KAPPA')
#plt.scatter(x_test_dim[:,1], y_svr_dim[:],  s=2, c='blue',    marker='+', label='Support Vector Machine')
#plt.scatter(x_test_dim[:,1], y_kr_dim[:],   s=2, c='green',   marker='p', label='Kernel Ridge')
#plt.scatter(x_test_dim[:,1], y_rf_dim[:],   s=2, c='cyan',    marker='*', label='Random Forest')
plt.scatter(x_test_dim[:,1], y_kn_dim[:],   s=2, c='magenta', marker='d', label='k-Nearest Neighbour')
#plt.scatter(x_test_dim[:,1], y_gp_dim[:],   s=2, c='orange',  marker='^', label='Gaussian Process')
#plt.scatter(x_test_dim[:,1], y_sgd_dim[:],  s=2, c='yellow',  marker='*', label='Stochastic Gradient Descent')
plt.title('Thermal conductivity regression with kNN')
plt.ylabel(r'$\lambda$ [W/m·K]')
plt.xlabel('T [K] ')
plt.legend()
plt.tight_layout()
plt.savefig("lambda_kNN.pdf", dpi=150, crop='false')
plt.show()
