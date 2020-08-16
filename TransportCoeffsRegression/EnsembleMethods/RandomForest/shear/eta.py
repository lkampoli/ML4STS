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

from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

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
X = sc_x.fit_transform(x)
Y = sc_y.fit_transform(y)

# The data is then split into training and test data
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.75, test_size=0.25, random_state=42)

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', x_test.shape)
print('Testing Labels Shape:', y_test.shape)

# Random Forest
hyper_params = [{'n_estimators': (10, 100, 1000),
                 'min_weight_fraction_leaf': (0.0, 0.25, 0.5),
                 'max_features': ('sqrt','log2',None),
}]

est=ensemble.RandomForestRegressor()
grid_clf = GridSearchCV(est, cv=5, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2')

#rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

t0 = time.time()
grid_clf.fit(x_train, y_train.ravel())
runtime = time.time() - t0
print("RF complexity and bandwidth selected and model fitted in %.3f s" % runtime)

#t0 = time.time()
#y_rf = rf.predict(x_test)
#rf_predict = time.time() - t0
#print("RF prediction for %d inputs in %.3f s" % (x_test.shape[0], rf_predict))

#print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_rf))
#print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_rf))
#print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_rf)))

train_score_mse  = mean_squared_error(sc_y.inverse_transform(y_train), sc_y.inverse_transform(grid_clf.predict(x_train)))
train_score_mae  = mean_absolute_error(sc_y.inverse_transform(y_train),sc_y.inverse_transform(grid_clf.predict(x_train)))
train_score_evs  = explained_variance_score(sc_y.inverse_transform(y_train), sc_y.inverse_transform(grid_clf.predict(x_train)))
train_score_me   = max_error(sc_y.inverse_transform(y_train), sc_y.inverse_transform(grid_clf.predict(x_train)))

test_score_mse  = mean_squared_error(sc_y.inverse_transform(y_test),  sc_y.inverse_transform(grid_clf.predict(x_test)))
test_score_mae  = mean_absolute_error(sc_y.inverse_transform(y_test), sc_y.inverse_transform(grid_clf.predict(x_test)))
test_score_evs  = explained_variance_score(sc_y.inverse_transform(y_test),  sc_y.inverse_transform(grid_clf.predict(x_test)))
test_score_me   = max_error(sc_y.inverse_transform(y_test),  sc_y.inverse_transform(grid_clf.predict(x_test)))

sorted_grid_params = sorted(grid_clf.best_params_.items(), key=operator.itemgetter(0))

out_text = '\t'.join(['random-forest',
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

best_n_estimators = grid_clf.best_params_['n_estimators']
best_min_weight_fraction_leaf = grid_clf.best_params_['min_weight_fraction_leaf']
best_max_features = grid_clf.best_params_['max_features']

# open a (new) file to write
outF = open("output.txt", "w")
print('best_n_estimators = ', best_n_estimators, file=outF)
print('best_min_weight_fraction_leaf = ', best_min_weight_fraction_leaf, file=outF)
print('best_max_features = ', best_max_features, file=outF)
outF.close()

rf = RandomForestRegressor(n_estimators=best_n_estimators,
                           min_weight_fraction_leaf=best_min_weight_fraction_leaf,
                           max_features=best_max_features)

t0 = time.time()
rf.fit(x_train, y_train.ravel())
rf_fit = time.time() - t0
print("RF complexity and bandwidth selected and model fitted in %.3f s" % rf_fit)

t0 = time.time()
y_rf = rf.predict(x_test)
rf_predict = time.time() - t0
print("RF prediction for %d inputs in %.3f s" % (x_test.shape[0], rf_predict))

# open a file to append
outF = open("output.txt", "a")
print("RF complexity and bandwidth selected and model fitted in %.3f s" % rf_fit, file=outF)
print("RF prediction for %d inputs in %.3f s" % (x_test.shape[0], rf_predict),file=outF)
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_rf), file=outF)
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_rf), file=outF)
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_rf)), file=outF)
outF.close()

print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_rf))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_rf))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_rf)))

x_test_dim = sc_x.inverse_transform(x_test)
y_test_dim = sc_y.inverse_transform(y_test)
y_rf_dim   = sc_y.inverse_transform(y_rf)

plt.scatter(x_test_dim[:,1], y_test_dim[:], s=5, c='red',     marker='o', label='KAPPA')
#plt.scatter(x_test_dim[:,1], y_svr_dim[:],  s=2, c='blue',    marker='+', label='Support Vector Machine')
#plt.scatter(x_test_dim[:,1], y_kr_dim[:],   s=2, c='green',   marker='p', label='Kernel Ridge')
plt.scatter(x_test_dim[:,1], y_rf_dim[:],   s=2, c='cyan',    marker='*', label='Random Forest')
#plt.scatter(x_test_dim[:,1], y_kn_dim[:],   s=2, c='magenta', marker='d', label='k-Nearest Neighbour')
#plt.scatter(x_test_dim[:,1], y_gp_dim[:],   s=2, c='orange',  marker='^', label='Gaussian Process')
#plt.scatter(x_test_dim[:,1], y_sgd_dim[:],  s=2, c='yellow',  marker='*', label='Stochastic Gradient Descent')
#plt.title('Shear viscosity regression with SVR, KR, RF, kN, GP')
plt.title('Shear viscosity regression with RF')
plt.ylabel(r'$\eta$ [Pa·s]')
plt.xlabel('T [K] ')
plt.legend()
plt.tight_layout()
plt.savefig("eta_RF.pdf", dpi=150, crop='false')
plt.show()
