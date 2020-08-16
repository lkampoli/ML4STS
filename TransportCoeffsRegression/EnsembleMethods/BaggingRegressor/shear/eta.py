#!/usr/bin/env python

# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html#sklearn.ensemble.BaggingRegressor
# https://scikit-learn.org/stable/modules/ensemble.html#bagging
# https://scikit-learn.org/stable/auto_examples/ensemble/plot_bias_variance.html#sphx-glr-auto-examples-ensemble-plot-bias-variance-py

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
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVR

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

# The data is split into training and test
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.75, test_size=0.25, random_state=42)

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:',   y_train.shape)
print('Testing Features Shape:',  x_test.shape)
print('Testing Labels Shape:',    y_test.shape)

hyper_params = [{'n_estimators': (10, 100, 1000,),
                 'base_estimator': (SVR(),None,),
                 'max_samples': (1,10,100,1000,),
                 'max_features': (1,10,100,),
                 'bootstrap': (True, False,),
                 'bootstrap_features': (True, False,),
                 'oob_score': (True, False,),
                 'warm_start': (True, False,),
}]

est=ensemble.BaggingRegressor()
gs = GridSearchCV(est, cv=5, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2')

t0 = time.time()
gs.fit(x_train, y_train.ravel())
runtime = time.time() - t0
print("B complexity and bandwidth selected and model fitted in %.6f s" % runtime)

train_score_mse = mean_squared_error(      sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_mae = mean_absolute_error(     sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_evs = explained_variance_score(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_me  = max_error(               sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))

test_score_mse  = mean_squared_error(      sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))
test_score_mae  = mean_absolute_error(     sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))
test_score_evs  = explained_variance_score(sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))
test_score_me   = max_error(               sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))

sorted_grid_params = sorted(gs.best_params_.items(), key=operator.itemgetter(0))

out_text = '\t'.join(['bagging',
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

best_n_estimators = gs.best_params_['n_estimators']
best_base_estimator = gs.best_params_['base_estimator']
best_max_samples = gs.best_params_['max_samples']
best_max_features = gs.best_params_['max_features']
best_bootstrap = gs.best_params_['bootstrap']
best_bootstrap_features = gs.best_params_['bootstrap_features']
best_oob_score = gs.best_params_['oob_score']
best_warm_start = gs.best_params_['warm_start']

outF = open("output.txt", "w")
print('best_n_estimators = ', best_n_estimators, file=outF)
print('best_base_estimator = ', best_base_estimator, file=outF)
print('best_max_samples = ', best_max_samples, file=outF)
print('best_max_features = ', best_max_features, file=outF)
print('best_bootstrap = ', best_bootstrap, file=outF)
print('best_oob_score = ', best_oob_score, file=outF)
print('best_warm_start = ', best_warm_start, file=outF)
outF.close()

regr = BaggingRegressor(base_estimator=best_base_estimator,
                        n_estimators=best_n_estimators,
                        max_samples=best_max_samples,
                        max_features=best_max_features,
                        bootstrap=best_bootstrap,
                        oob_score=best_oob_score,
                        warm_start=best_warm_start,
                        random_state=0)

t0 = time.time()
regr.fit(x_train, y_train.ravel())
regr_fit = time.time() - t0
print("B complexity and bandwidth selected and model fitted in %.6f s" % regr_fit)

t0 = time.time()
y_regr = regr.predict(x_test)
regr_predict = time.time() - t0
print("B prediction for %d inputs in %.6f s" % (x_test.shape[0], regr_predict))

regr.score(x_train, y_train)

outF = open("output.txt", "a")
print("B complexity and bandwidth selected and model fitted in %.6f s" % regr_fit,             file=outF)
print("B prediction for %d inputs in %.6f s" % (x_test.shape[0], regr_predict),                file=outF)
print('Mean Absolute Error (MAE):',              metrics.mean_absolute_error(y_test, y_regr),  file=outF)
print('Mean Squared Error (MSE):',               metrics.mean_squared_error( y_test, y_regr),  file=outF)
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error( y_test, y_regr)), file=outF)
print('Score:', regr.score(x_train, y_train),                                                  file=outF)
outF.close()

print('Mean Absolute Error (MAE):',              metrics.mean_absolute_error(y_test, y_regr))
print('Mean Squared Error (MSE):',               metrics.mean_squared_error( y_test, y_regr))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error( y_test, y_regr)))

x_test_dim = sc_x.inverse_transform(x_test)
y_test_dim = sc_y.inverse_transform(y_test)
y_regr_dim = sc_y.inverse_transform(y_regr)

plt.scatter(x_test_dim[:,1], y_test_dim[:], s=5, c='r', marker='o', label='KAPPA')
plt.scatter(x_test_dim[:,1], y_regr_dim[:], s=2, c='k', marker='*', label='Bagging')
plt.title('Shear viscosity regression with B')
plt.ylabel(r'$\eta$ [Pa·s]')
plt.xlabel('T [K] ')
plt.legend()
plt.tight_layout()
plt.savefig("eta_B.pdf", dpi=150, crop='false')
plt.show()