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
from joblib import dump, load
import pickle

n_jobs = -1
trial  = 1

#dataset=np.loadtxt("../data/datarelax.txt")
dataset=np.loadtxt("../data/datasetDR.txt")
#dataset=np.loadtxt("../data/datasetVT.txt")
#dataset=np.loadtxt("../data/datasetVV.txt")

x = dataset[:,2:3]   # 0: x [m], 1: t [s], 2: T [K]
y = dataset[:,9:10]  # Rci (relaxation source terms)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=69)

sc_x = StandardScaler()
sc_y = StandardScaler()

# fit scaler
sc_x.fit(x_train)
# transform training datasetx
x_train = sc_x.transform(x_train)
# transform test dataset
x_test = sc_x.transform(x_test)

# fit scaler on training dataset
sc_y.fit(y_train)
# transform training dataset
y_train = sc_y.transform(y_train)
# transform test dataset
y_test = sc_y.transform(y_test)

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', x_test.shape)
print('Testing Labels Shape:', y_test.shape)

# Extra Trees
hyper_params = [{'n_estimators': (10, 100, 1000,),
                 'min_weight_fraction_leaf': (0.0, 0.25, 0.5,),
#                 'max_features': ('sqrt','log2','auto', None,),
#                 'max_samples': (1,10,100,1000,),
#                 'bootstrap': (True, False,),
#                 'oob_score': (True, False,),
#                 'warm_start': (True, False,),
#                 'criterion': ('mse', 'mae',),
                 'max_depth': (1,10,100,None,),
                 'max_leaf_nodes': (2,10,100,),
                 'min_samples_split': (2,10,100,), #0.1,0.25,0.5,0.75,1.0,),
                 'min_samples_leaf': (1,10,100,),
}]

est=ensemble.ExtraTreesRegressor()
gs = GridSearchCV(est, cv=5, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2')

t0 = time.time()
gs.fit(x_train, y_train.ravel())
runtime = time.time() - t0
print("Complexity and bandwidth selected and model fitted in %.6f s" % runtime)

train_score_mse = mean_squared_error(      sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_mae = mean_absolute_error(     sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_evs = explained_variance_score(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_me  = max_error(               sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))

test_score_mse  = mean_squared_error(      sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))
test_score_mae  = mean_absolute_error(     sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))
test_score_evs  = explained_variance_score(sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))
test_score_me   = max_error(               sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))

sorted_grid_params = sorted(gs.best_params_.items(), key=operator.itemgetter(0))

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

# ExtraTrees
best_n_estimators = gs.best_params_['n_estimators']
best_min_weight_fraction_leaf = gs.best_params_['min_weight_fraction_leaf']
#best_max_features = gs.best_params_['max_features']
#best_max_samples = gs.best_params_['max_samples']
#best_bootstrap = gs.best_params_['bootstrap']
#best_oob_score = gs.best_params_['oob_score']
#best_warm_start = gs.best_params_['warm_start']
#best_criterion = gs.best_params_['criterion']
best_max_depth = gs.best_params_['max_depth']
best_min_samples_split = gs.best_params_['min_samples_split']
best_min_samples_leaf = gs.best_params_['min_samples_leaf']
best_max_leaf_nodes = gs.best_params_['max_leaf_nodes']

outF = open("output.txt", "w")
print('best_n_estimators = ', best_n_estimators, file=outF)
print('best_min_weight_fraction_leaf = ', best_min_weight_fraction_leaf, file=outF)
#print('best_max_features = ', best_max_features, file=outF)
#print('best_bootstrap = ', best_bootstrap, file=outF)
#print('best_oob_score = ', best_oob_score, file=outF)
#print('best_warm_start = ', best_warm_start, file=outF)
#print('best_criterion = ', best_criterion, file=outF)
print('best_max_depth = ', best_max_depth, file=outF)
print('best_min_samples_split = ', best_min_samples_split, file=outF)
print('best_min_samples_leaf = ', best_min_samples_leaf, file=outF)
#ddprint('best_max_samples = ', best_max_samples, file=outF)
print('best_max_leaf_nodes = ', best_max_leaf_nodes, file=outF)
outF.close()

regr = ExtraTreesRegressor(n_estimators=best_n_estimators,
                           min_weight_fraction_leaf=best_min_weight_fraction_leaf,
#                           max_features=best_max_features,
#                           bootstrap=best_bootstrap,
#                           oob_score=best_oob_score,
#                           warm_start=best_warm_start,
#                           criterion=best_criterion,
#                           max_samples=best_max_samples,
                           max_depth=best_max_depth,
                           max_leaf_nodes=best_max_leaf_nodes,
                           min_samples_split=best_min_samples_split,
                           min_samples_leaf=best_min_samples_leaf)

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

plt.scatter(x_test_dim, y_test_dim, s=2, c='k', marker='o', label='Matlab')
plt.scatter(x_test_dim, y_regr_dim, s=2, c='r', marker='+', label='ExtraTrees')
#plt.title('Relaxation term $R_{ci}$ regression')
plt.ylabel('$R_{ci}$ $[J/m^3/s]$')
plt.xlabel('T [K] ')
plt.legend()
plt.tight_layout()
plt.savefig("regression_ET.eps", dpi=150, crop='false')
plt.savefig("regression_ET.pdf", dpi=150, crop='false')
plt.show()

# save the model to disk
dump(gs, 'model_ET.sav')
