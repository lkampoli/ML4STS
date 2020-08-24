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
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load
import pickle
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

n_jobs = -1
trial  = 1

dataset=np.loadtxt("../data/datarelax.txt")

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

#y=np.reshape(y, (-1,1))
#sc_x = StandardScaler()
#sc_y = StandardScaler()
#X = sc_x.fit_transform(x)
#Y = sc_y.fit_transform(y)

#x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.75, test_size=0.25, random_state=42)
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

# Random Forest
hyper_params = [{
                 'n_estimators': (1, 10, 100, 1000,),
                 #'n_estimators': (1, 10, 100, 1000, 10000),
                 'min_weight_fraction_leaf': (0.0, 0.01, 0.02,),
                 #'min_weight_fraction_leaf': (0.0, 0.25, 0.5, 0.75, 0.9),
                 'max_features': ('auto','sqrt','log2'),
                 'criterion': ('mse', 'mae'),
                  #'max_samples': (0.1, 0.25, 0.5, 0.75, 0.9,),
                  #'bootstrap': (True, False),
                  #'warm_start': (True, False),
                  #'min_impurity_decrease': (0.1, 0.25, 0.5. 0.75, 0.9,),
                  #'max_leaf_nodes': (1, 10, 100, 1000,),
}]

est=ensemble.RandomForestRegressor()

gs = GridSearchCV(est, cv=10, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2')

t0 = time.time()
gs.fit(x_train, y_train.ravel())
runtime = time.time() - t0
print("Complexity and bandwidth selected and model fitted in %.6f s" % runtime)

# save the model to disk
dump(gs, 'model.sav')

train_score_mse = mean_squared_error(      sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_mae = mean_absolute_error(     sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_evs = explained_variance_score(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_me  = max_error(               sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))

test_score_mse  = mean_squared_error(      sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))
test_score_mae  = mean_absolute_error(     sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))
test_score_evs  = explained_variance_score(sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))
test_score_me   = max_error(               sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))
test_score_msle = mean_squared_log_error(  sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))
test_score_r2   = r2_score(                sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))

print("The model performance for testing set")
print("--------------------------------------")
print('MAE is {}'.format(test_score_mae))
print('MSE is {}'.format(test_score_mse))
print('MSLE is {}'.format(test_score_msle))
print('EVS is {}'.format(test_score_evs))
print('ME is {}'.format(test_score_me))
print('R2 score is {}'.format(test_score_r2))

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
best_n_estimators = gs.best_params_['n_estimators']
best_min_weight_fraction_leaf = gs.best_params_['min_weight_fraction_leaf']
best_max_features = gs.best_params_['max_features']
best_criterion = gs.best_params_['criterion']

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
print('best_n_estimators = ', best_n_estimators, file=outF)
print('best_min_weight_fraction_leaf = ', best_min_weight_fraction_leaf, file=outF)
print('best_max_features = ', best_max_features, file=outF)
print('best_criterion = ', best_criterion, file=outF)
outF.close()

#regr = KNeighborsRegressor(n_neighbors=best_n_neighbors, algorithm=best_algorithm,
#                         leaf_size=best_leaf_size, weights=best_weights, p=best_p)
#regr = KernelRidge(kernel=best_kernel, gamma=best_gamma, alpha=best_alpha)
regr = RandomForestRegressor(n_estimators=best_n_estimators,
                             min_weight_fraction_leaf=best_min_weight_fraction_leaf,
                             max_features=best_max_features,
                             criterion=best_criterion,
                             random_state=69)

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
plt.scatter(x_test_dim, y_regr_dim, s=2, c='k', marker='d', label='RandomForest')
#plt.title('Relaxation term $R_{ci}$ regression')
plt.ylabel('$R_{ci}$ $[J/m^3/s]$')
plt.xlabel('T [K] ')
plt.legend()
plt.tight_layout()
plt.savefig("regression_RF.eps", dpi=150, crop='false')
plt.savefig("regression_RF.pdf", dpi=150, crop='false')
plt.show()
