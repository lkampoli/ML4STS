#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

#from sklearn.svm import SVR

#from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import MinMaxScaler, StandardScaler

#from sklearn.linear_model import LinearRegression, SGDRegressor

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, cross_val_score

from sklearn import kernel_ridge
from sklearn.kernel_ridge import KernelRidge

#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, DotProduct, RBF, RationalQuadratic, ConstantKernel

#from sklearn.tree import DecisionTreeRegressor

#from sklearn.ensemble import RandomForestRegressor

#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.neighbors import RadiusNeighborsRegressor

n_jobs = 1
trial  = 1

# Import database
dataset=np.loadtxt("../../data/dataset_TD.csv", delimiter=",")
x=dataset[:,0:3]
y=dataset[:,3] # 0: X, 1: T, 2: shear, 3: bulk, 4: conductivity

# Plot dataset
#plt.scatter(x[:,1], dataset[:,2], s=0.5)
#plt.title('Shear viscosity')
#plt.xlabel('T [K]')
#plt.ylabel(r'$\eta$')
#plt.show()

#plt.scatter(x[:,1], dataset[:,3], s=0.5)
#plt.title('Bulk viscosity')
#plt.xlabel('T [K]')
#plt.ylabel(r'$\zeta$')
#plt.show()

#plt.scatter(x[:,1], dataset[:,4], s=0.5)
#plt.title('Thermal conductivity')
#plt.xlabel('T [K]')
#plt.ylabel(r'$\lambda$')
#plt.show()

y=np.reshape(y, (-1,1))
sc_x = StandardScaler()
sc_y = StandardScaler()
#sc_x = MinMaxScaler()
#sc_y = MinMaxScaler()
X = sc_x.fit_transform(x)
Y = sc_y.fit_transform(y)
#y=np.reshape(y, (-1,1))
#sc_x = MinMaxScaler()
#sc_y = MinMaxScaler()
#print(sc_x.fit(x))
#X=sc_x.transform(x)
#print(sc_y.fit(y))
#Y=sc_y.transform(y)

# The data is then split into training and test data
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.75, test_size=0.25, random_state=42)
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', x_test.shape)
print('Testing Labels Shape:', y_test.shape)

# Fit regression model

# gamma{‘scale’, ‘auto’} or float, default=’scale’
#
#    Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
#        if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma,
#        if ‘auto’, uses 1 / n_features.

# Cfloat, default=1.0
#
#    Regularization parameter. The strength of the regularization is inversely proportional to C.
#    Must be strictly positive. The penalty is a squared l2 penalty.

# epsilonfloat, default=0.1

#    Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is
#    associated in the training loss function with points predicted within a distance epsilon from the
#    actual value.

# SVR
#pipeline = make_pipeline(preprocessing.StandardScaler(), SVR(kernel='rbf', epsilon=0.01, C=100, gamma = 0.01))
#pipeline = make_pipeline(SVR(kernel='rbf', epsilon=0.01, C=100, gamma = 'scale'))
#pipeline = make_pipeline(SVR(kernel='linear', C=100, gamma='auto'))
#pipeline = make_pipeline(SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1, coef0=1))
#svr = SVR(kernel='rbf', epsilon=0.01, C=100, gamma = 'auto')
#svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1),
#                  param_grid={"C": [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
#                               "epsilon": [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
#                               "gamma": np.logspace(-2, 2, 5)})

# KRR
hyper_params = [{'kernel': ('poly','rbf',), 'alpha': (1e-4,1e-2,0.1,1,10,), 'gamma': (0.01,0.1,1,10,100,),}]
#hyper_params = [{'kernel': ('poly','rbf','sigmoid',), 'alpha': (1e-4,1e-2,0.1,1,), 'gamma': (0.01,0.1,1,10,),}]

est=kernel_ridge.KernelRidge()

#kr = KernelRidge(kernel='rbf', gamma=0.1)
#kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)})
grid_clf = GridSearchCV(est, cv=5, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2')

# Random Forest
#rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

# k-Nearest Neighbour
#kn = KNeighborsRegressor(n_neighbors=1, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski',
#                         metric_params=None, n_jobs=None)

# Gaussian Process
#gp_kernel = ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) + WhiteKernel(1e-1)
###gp_kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
#gp_kernel = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
#gp_kernel = 1.0 * RBF(1.0)
###gp = GaussianProcessRegressor(kernel=gp_kernel)

# Stochastic Gradient Descent
#sgd = SGDRegressor(max_iter=1000, tol=1e-3)

#t0 = time.time()
#svr.fit(x_train, y_train)
#svr_fit = time.time() - t0
#print("SVR complexity and bandwidth selected and model fitted in %.3f s" % svr_fit)

t0 = time.time()
# fit model
grid_clf.fit(x_train, y_train.ravel())
#kr.fit(x_train, y_train)
runtime = time.time() - t0
print("KRR complexity and bandwidth selected and model fitted in %.3f s" % runtime)

#t0 = time.time()
#rf.fit(x_train, y_train)
#rf_fit = time.time() - t0
#print("RF complexity and bandwidth selected and model fitted in %.3f s" % rf_fit)

#t0 = time.time()
#kn.fit(x_train, y_train)
#kn_fit = time.time() - t0
#print("KN complexity and bandwidth selected and model fitted in %.3f s" % kn_fit)

#t0 = time.time()
#gp.fit(x_train, y_train)
#gp_fit = time.time() - t0
#print("GP complexity and bandwidth selected and model fitted in %.3f s" % gp_fit)

#t0 = time.time()
#sgd.fit(x_train, y_train)
#sgd_fit = time.time() - t0
#print("SGD complexity and bandwidth selected and model fitted in %.3f s" % sgd_fit)

#t0 = time.time()
#y_svr = svr.predict(x_test)
#svr_predict = time.time() - t0
#print("SVR prediction for %d inputs in %.3f s" % (x_test.shape[0], svr_predict))

#print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_svr))
#print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_svr))
#print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_svr)))

#t0 = time.time()
#y_kr = kr.predict(x_test)
#kr_predict = time.time() - t0
#print("KRR prediction for %d inputs in %.3f s" % (x_test.shape[0], kr_predict))

#print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_kr))
#print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_kr))
#print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_kr)))

#t0 = time.time()
#y_rf = rf.predict(x_test)
#rf_predict = time.time() - t0
#print("RF prediction for %d inputs in %.3f s" % (x_test.shape[0], rf_predict))

#print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_rf))
#print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_rf))
#print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_rf)))

#t0 = time.time()
#y_kn = kn.predict(x_test)
#kn_predict = time.time() - t0
#print("KN prediction for %d inputs in %.3f s" % (x_test.shape[0], kn_predict))

#print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_kn))
#print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_kn))
#print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_kn)))

#t0 = time.time()
#y_gp, sigma = gp.predict(x_test, return_std=True)
#gp_predict = time.time() - t0
#print("GP prediction for %d inputs in %.3f s" % (x_test.shape[0], gp_predict))

#print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_gp))
#print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_gp))
#print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_gp)))

#t0 = time.time()
#y_sgd = sgd.predict(x_test)
#sgd_predict = time.time() - t0
#print("SGD prediction for %d inputs in %.3f s" % (x_test.shape[0], sgd_predict))

#print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_sgd))
#print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_sgd))
#print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_sgd)))

#x_test_dim = sc_x.inverse_transform(x_test)
#y_test_dim = sc_y.inverse_transform(y_test)
#y_svr_dim  = sc_y.inverse_transform(y_svr)
#y_kr_dim   = sc_y.inverse_transform(y_kr)
#y_rf_dim   = sc_y.inverse_transform(y_rf)
#y_kn_dim   = sc_y.inverse_transform(y_kn)
#y_gp_dim   = sc_y.inverse_transform(y_gp)
#y_sgd_dim  = sc_y.inverse_transform(y_sgd)

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

#r2_score(y_true, y_pred, multioutput='variance_weighted')
#r2_score(y_true, y_pred, multioutput='uniform_average')
#r2_score(y_true, y_pred, multioutput='raw_values')

sorted_grid_params = sorted(grid_clf.best_params_.items(), key=operator.itemgetter(0))

out_text = '\t'.join(['kernel-ridge',
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

best_kernel = grid_clf.best_params_['kernel']
best_alpha  = grid_clf.best_params_['alpha']
best_gamma  = grid_clf.best_params_['gamma']

# open a (new) file to write
outF = open("output.txt", "w")
print('best_kernel = ', best_kernel, file=outF)
print('best_alpha = ', best_alpha, file=outF)
print('best_gamma = ', best_gamma, file=outF)
outF.close()

best_kernel = grid_clf.best_params_['kernel']
best_alpha  = grid_clf.best_params_['alpha']
best_gamma  = grid_clf.best_params_['gamma']

kr = KernelRidge(kernel=best_kernel, gamma=best_gamma, alpha=best_alpha)

t0 = time.time()
kr.fit(x_train, y_train.ravel())
kr_fit = time.time() - t0
print("KR complexity and bandwidth selected and model fitted in %.3f s" % kr_fit)

t0 = time.time()
y_kr = kr.predict(x_test)
kr_predict = time.time() - t0
print("KR prediction for %d inputs in %.3f s" % (x_test.shape[0], kr_predict))

# open a file to append
outF = open("output.txt", "a")
print("KR complexity and bandwidth selected and model fitted in %.3f s" % kr_fit, file=outF)
print("KR prediction for %d inputs in %.3f s" % (x_test.shape[0], kr_predict),file=outF)
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_kr), file=outF)
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_kr), file=outF)
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_kr)), file=outF)
outF.close()

print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_kr))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_kr))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_kr)))

x_test_dim = sc_x.inverse_transform(x_test)
y_test_dim = sc_y.inverse_transform(y_test)
y_kr_dim   = sc_y.inverse_transform(y_kr)

plt.scatter(x_test_dim[:,1], y_test_dim[:], s=5, c='red',     marker='o', label='KAPPA')
#plt.scatter(x_test_dim[:,1], y_svr_dim[:],  s=2, c='blue',    marker='+', label='Support Vector Machine')
plt.scatter(x_test_dim[:,1], y_kr_dim[:],   s=2, c='green',   marker='p', label='Kernel Ridge')
#plt.scatter(x_test_dim[:,1], y_rf_dim[:],   s=2, c='cyan',    marker='*', label='Random Forest')
#plt.scatter(x_test_dim[:,1], y_kn_dim[:],   s=2, c='magenta', marker='d', label='k-Nearest Neighbour')
#plt.scatter(x_test_dim[:,1], y_gp_dim[:],   s=2, c='orange',  marker='^', label='Gaussian Process')
#plt.scatter(x_test_dim[:,1], y_sgd_dim[:],  s=2, c='yellow',  marker='*', label='Stochastic Gradient Descent')
plot.title('Thermal diffusion regression with KR')
plt.ylabel(r'$D_T$ $[m^2/s]$')
plt.xlabel('T [K] ')
plt.legend()
plt.tight_layout()
plt.savefig("TD_KR.pdf", dpi=150, crop='false')
plt.show()

