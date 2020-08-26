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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import kernel_ridge
from sklearn.kernel_ridge import KernelRidge
from joblib import dump, load
import pickle
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

n_jobs = -1
trial  = 1

#dataset=np.loadtxt("../data/datarelax.txt")
dataset=np.loadtxt("../data/datasetDR.txt")
#dataset=np.loadtxt("../data/datasetVT.txt")
#dataset=np.loadtxt("../data/datasetVV.txt")
x=dataset[:,2:3]   # 0: x [m], 1: t [s], 2: T [K]
y=dataset[:,9:10]  # Rci (relaxation source terms)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=69, shuffle=True)

# https://stackoverflow.com/questions/43675665/when-scale-the-data-why-the-train-dataset-use-fit-and-transform-but-the-te
# https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/

#sc_x = MinMaxScaler(feature_range=(0, 1))
sc_x = StandardScaler()
sc_y = StandardScaler()

# fit scaler
sc_x.fit(x_train)
# transform training datasetx
x_train = sc_x.transform(x_train)
# transform test dataset
x_test = sc_x.transform(x_test)

#y_train = y_train.reshape(len(y_train), 1)
#y_test = y_test.reshape(len(y_train), 1)

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

hyper_params = [{'kernel': ('poly', 'rbf',),
                 'alpha': (1e-3, 1e-2, 1e-1, 0.0, 0.25, 0.5, 0.75, 1.,),
                 'gamma': (0.1, 1, 10, 100, 1000,),}]

est=kernel_ridge.KernelRidge()
gs = GridSearchCV(est, cv=10, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2')

t0 = time.time()
gs.fit(x_train, y_train.ravel())
runtime = time.time() - t0
print("KR complexity and bandwidth selected and model fitted in %.6f s" % runtime)

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

best_kernel = gs.best_params_['kernel']
best_alpha  = gs.best_params_['alpha']
best_gamma  = gs.best_params_['gamma']

outF = open("output.txt", "w")
print('best_kernel = ', best_kernel, file=outF)
print('best_alpha = ', best_alpha, file=outF)
print('best_gamma = ', best_gamma, file=outF)
outF.close()

regr = KernelRidge(kernel=best_kernel, gamma=best_gamma, alpha=best_alpha)

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
print("KR complexity and bandwidth selected and model fitted in %.6f s" % regr_fit, file=outF)
print("KR prediction for %d inputs in %.6f s" % (x_test.shape[0], regr_predict),file=outF)
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
plt.scatter(x_test_dim, y_regr_dim, s=2, c='r', marker='+', label='KernelRidge')
#plt.title('Relaxation term $R_{ci}$ regression')
plt.ylabel('$R_{ci}$ $[J/m^3/s]$')
plt.xlabel('T [K] ')
plt.legend()
plt.tight_layout()
plt.savefig("regression_KR.eps", dpi=150, crop='false')
plt.savefig("regression_KR.pdf", dpi=150, crop='false')
plt.show()

# save the model to disk
dump(gs, 'model_KR.sav')
