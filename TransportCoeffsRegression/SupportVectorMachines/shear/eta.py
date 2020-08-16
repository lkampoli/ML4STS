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
from sklearn import svm
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

n_jobs = 1
trial  = 1

# Import database
dataset=np.loadtxt("../../data/dataset_lite.csv", delimiter=",")
x=dataset[:,0:2]
y=dataset[:,2] # 0: X, 1: T, 2: shear, 3: bulk, 4: conductivity

# Plot dataset
#plt.scatter(x[:,1], dataset[:,2], s=0.5)
#plt.title('Shear viscosity')
#plt.xlabel('T [K]')
#plt.ylabel(r'$\eta$')
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

#hyper_params = [{'kernel': ('linear', 'poly', 'rbf', 'sigmoid',), 'gamma': ('scale', 'auto',),
#                 'C': (1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3,), 'epsilon': (1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3,),
#                 'coef0': (0.,1.,10.,), }]

#hyper_params = [{'kernel': ('poly', 'rbf',), 'gamma': ('scale', 'auto',),
#                 'C': (1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3,), 'epsilon': (1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3,), }]

hyper_params = [{'kernel': ('poly', 'rbf',), 'gamma': ('scale', 'auto',),
                 'C': (1e-1, 1e0, 1e1,), 'epsilon': (1e-1, 1e0, 1e1,), }]

est=svm.SVR()

grid_clf = GridSearchCV(est, cv=5, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2')

t0 = time.time()
grid_clf.fit(x_train, y_train.ravel())
runtime = time.time() - t0
print("SVR complexity and bandwidth selected and model fitted in %.3f s" % runtime)

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

out_text = '\t'.join(['svr',
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
best_gamma = grid_clf.best_params_['gamma']
best_C = grid_clf.best_params_['C']
best_epsilon = grid_clf.best_params_['epsilon']
#best_coef0 = grid_clf.best_params_['coef0']

# open a (new) file to write
outF = open("output.txt", "w")
print('best_kernel = ', best_kernel, file=outF)
print('best_gamma = ', best_gamma, file=outF)
print('best_C = ', best_C, file=outF)
print('best_epsilon = ', best_epsilon, file=outF)
#print('best_coef0 = ', best_coef0, file=outF)
outF.close()

#svr = SVR(kernel=best_kernel, epsilon=best_epsilon, C=best_C, gamma=best_gamma, coef0=best_coef0)
svr = SVR(kernel=best_kernel, epsilon=best_epsilon, C=best_C, gamma=best_gamma)

t0 = time.time()
svr.fit(x_train, y_train.ravel())
svr_fit = time.time() - t0
print("SVR complexity and bandwidth selected and model fitted in %.3f s" % svr_fit)

t0 = time.time()
y_svr = svr.predict(x_test)
svr_predict = time.time() - t0
print("SVR prediction for %d inputs in %.3f s" % (x_test.shape[0], svr_predict))

# open a file to append
outF = open("output.txt", "a")
print("SVR complexity and bandwidth selected and model fitted in %.3f s" % svr_fit, file=outF)
print("SVR prediction for %d inputs in %.3f s" % (x_test.shape[0], svr_predict),file=outF)
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_svr), file=outF)
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_svr), file=outF)
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_svr)), file=outF)
outF.close()

print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_svr))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_svr))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_svr)))

x_test_dim = sc_x.inverse_transform(x_test)
y_test_dim = sc_y.inverse_transform(y_test)
y_svr_dim   = sc_y.inverse_transform(y_svr)

plt.scatter(x_test_dim[:,1], y_test_dim[:], s=5, c='red',     marker='o', label='KAPPA')
plt.scatter(x_test_dim[:,1], y_svr_dim[:],  s=2, c='blue',    marker='+', label='Support Vector Machine')
#plt.scatter(x_test_dim[:,1], y_kr_dim[:],   s=2, c='green',   marker='p', label='Kernel Ridge')
#plt.scatter(x_test_dim[:,1], y_rf_dim[:],   s=2, c='cyan',    marker='*', label='Random Forest')
#plt.scatter(x_test_dim[:,1], y_kn_dim[:],   s=2, c='magenta', marker='d', label='k-Nearest Neighbour')
#plt.scatter(x_test_dim[:,1], y_gp_dim[:],   s=2, c='orange',  marker='^', label='Gaussian Process')
#plt.scatter(x_test_dim[:,1], y_sgd_dim[:],  s=2, c='yellow',  marker='*', label='Stochastic Gradient Descent')
plt.title('Shear viscosity regression with SVR')
plt.ylabel(r'$\eta$ [Pa·s]')
plt.xlabel('T [K] ')
plt.legend()
plt.tight_layout()
plt.savefig("eta_SVR.pdf", dpi=150, crop='false')
plt.show()
