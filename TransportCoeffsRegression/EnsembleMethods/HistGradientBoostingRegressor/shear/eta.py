#!/usr/bin/env python

# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html#sklearn.ensemble.HistGradientBoostingRegressor
# https://scikit-learn.org/stable/modules/ensemble.html#monotonic-cst-gbdt

import time
import sys
sys.path.insert(0, '../../../../Utilities/')
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
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor

n_jobs = 1
trial  = 1

# Import database
dataset=np.loadtxt("../../../data/dataset_lite.csv", delimiter=",")
x=dataset[:,0:2]
y=dataset[:,2] # 0: X, 1: T, 2: shear, 3: bulk, 4: conductivity

# Plot dataset
#plt.scatter(x[:,1], dataset[:,2], s=0.5)
#plt.title('Shear viscosity')
#plt.xlabel('T [K]')
#plt.ylabel(r'$\eta$')
#plt.show()

y = np.reshape(y, (-1,1))
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

hyper_params = [{
#                 'n_estimators': (10, 100, 1000,),
#                 'learning_rate': (0.1,0.01,0.001,),
#                 'max_leaf_nodes': (None,),
#                 'l2_regularization': (0., 0.25, 0.5, 0.75, 1.0,),
#                 'monotonic_cst': ([1, -1, 0]),
#                 'min_weight_fraction_leaf': (0.0, 0.25, 0.5,),
#                 'max_features': ('sqrt','log2','auto', None,),
#                 'warm_start': (True, False,),
#                 'criterion': ('friedman_mse', 'mse', 'mae',),
#                 'max_depth': (1,10,100,None,),
#                 'min_samples_split': (0.1,0.25,0.5,0.75,1.0,),
                 'min_samples_leaf': (1,10,100,),
                 'loss': ('least_squares', 'least_absolute_deviation', 'poisson',),
}]

est=ensemble.HistGradientBoostingRegressor()
gs = GridSearchCV(est, cv=5, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2')

t0 = time.time()
gs.fit(x_train, y_train.ravel())
runtime = time.time() - t0
print("HGB complexity and bandwidth selected and model fitted in %.6f s" % runtime)

train_score_mse = mean_squared_error(      sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_mae = mean_absolute_error(     sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_evs = explained_variance_score(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_me  = max_error(               sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))

test_score_mse = mean_squared_error(      sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_mae = mean_absolute_error(     sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_evs = explained_variance_score(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_me  = max_error(               sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))

sorted_grid_params = sorted(gs.best_params_.items(), key=operator.itemgetter(0))

out_text = '\t'.join(['hist-gradient-boosting',
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

#best_n_estimators = gs.best_params_['n_estimators']
#best_min_weight_fraction_leaf = gs.best_params_['min_weight_fraction_leaf']
#best_max_features = gs.best_params_['max_features']
#best_warm_start = gs.best_params_['warm_start']
#best_max_depth = gs.best_params_['max_depth']
#best_criterion = gs.best_params_['criterion']
#best_min_samples_split = gs.best_params_['min_samples_split']
best_min_samples_leaf = gs.best_params_['min_samples_leaf']
#best_learning_rate = gs.best_params_['learning_rate']
best_max_leaf_nodes = gs.best_params_['max_leaf_nodes']
#best_l2_regularization = gs.best_params_['l2_regularization']
#best_loss = gs.best_params_['loss']

outF = open("output.txt", "w")
#print('best_n_estimators = ', best_n_estimators, file=outF)
#print('best_min_weight_fraction_leaf = ', best_min_weight_fraction_leaf, file=outF)
#print('best_max_features = ', best_max_features, file=outF)
#print('best_warm_start = ', best_warm_start, file=outF)
#print('best_max_depth = ', best_max_depth, file=outF)
#print('best_criterion = ', best_criterion, file=outF)
#print('best_min_samples_split = ', best_min_samples_split, file=outF)
print('best_min_samples_leaf = ', best_min_samples_leaf, file=outF)
#print('best_learning_rate = ', best_learning_rate, file=outF)
#print('best_max_leaf_nodes = ', best_max_leaf_nodes, file=outF)
#print('best_l2_regularization = ', best_l2_regularization, file=outF)
print('best_loss = ', best_loss, file=outF)
outF.close()

regr = HistGradientBoostingRegressor(loss=best_loss,
#                                    learning_rate=best_learning_rate,
                                     n_estimators=best_n_estimators,
#                                    min_weight_fraction_leaf=best_min_weight_fraction_leaf,
#                                    max_features=best_max_features,
#                                    warm_start=best_warm_start,
#                                    max_depth=best_max_depth,
#                                    max_leaf_nodes=best_max_leaf_nodes,
#                                    l2_regularization=best_l2_regularization,
#                                    criterion=best_criterion,
#                                    min_samples_split=best_min_samples_split,
                                     min_samples_leaf=best_min_samples_leaf)

t0 = time.time()
regr.fit(x_train, y_train.ravel())
regr_fit = time.time() - t0
print("HGB complexity and bandwidth selected and model fitted in %.6f s" % regr_fit)

t0 = time.time()
y_regr = regr.predict(x_test)
regr_predict = time.time() - t0
print("HGB prediction for %d inputs in %.6f s" % (x_test.shape[0], regr_predict))

# open a file to append
outF = open("output.txt", "a")
print("HGB complexity and bandwidth selected and model fitted in %.6f s" % regr_fit, file=outF)
print("HGB prediction for %d inputs in %.6f s" % (x_test.shape[0], regr_predict),file=outF)
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

plt.scatter(x_test_dim[:,1], y_test_dim[:], s=5, c='r', marker='o', label='KAPPA')
plt.scatter(x_test_dim[:,1], y_regr_dim[:], s=2, c='k', marker='*', label='Hist Gradient Boosting')
plt.title('Shear viscosity regression with HGB')
plt.ylabel(r'$\eta$ [Pa·s]')
plt.xlabel('T [K] ')
plt.legend()
plt.tight_layout()
plt.savefig("eta_HGB.pdf", dpi=150, crop='false')
plt.show()
