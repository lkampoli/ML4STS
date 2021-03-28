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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import neighbors
from sklearn.neighbors import KNeighborsRegressor
import pickle
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from joblib import dump, load

n_jobs = -1
trial = 1

#dataset_T = np.loadtxt("../data/N2-N2/dis/Temperatures.csv")
#dataset_k = np.loadtxt("../data/N2-N2/dis/DR_RATES-N2-N2-dis.csv")
#dataset   = "DR_RATES-N2-N2-dis"
dataset_T = np.loadtxt("../data/N2-N/dis/Temperatures.csv")
dataset_k = np.loadtxt("../data/N2-N/dis/DR_RATES-N2-N_-dis.csv")
dataset   = "DR_RATES-N2-N-dis"

Lev = sys.argv[1]

x = dataset_T.reshape(-1,1)  # T [K]
y = dataset_k[:,0+int(Lev):1+int(Lev)] # k_DR
#y = dataset_k[:,0:1]        # k_DR

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=69)

sc_x = StandardScaler()
sc_y = StandardScaler()

sc_x.fit(x_train)
x_train = sc_x.transform(x_train)
x_test = sc_x.transform(x_test)

sc_y.fit(y_train)
y_train = sc_y.transform(y_train)
y_test = sc_y.transform(y_test)

dump(sc_x, open('../scaler/scaler_x_'+dataset+'_'+Lev+'.pkl', 'wb'))
dump(sc_y, open('../scaler/scaler_y_'+dataset+'_'+Lev+'.pkl', 'wb'))

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', x_test.shape)
print('Testing Labels Shape:', y_test.shape)

# kNN
hyper_params = [{'algorithm': ('ball_tree', 'kd_tree', 'brute',),
                 'n_neighbors': (1,2,3,4,5,6,7,8,9,10,),
                 'leaf_size': (1, 10, 20, 30, 100, 1000,),
                 'weights': ('uniform', 'distance',),
                 'p': (1,2,),}]

est=neighbors.KNeighborsRegressor()
gs = GridSearchCV(est, cv=10, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2')

t0 = time.time()
gs.fit(x_train, y_train.ravel())
runtime = time.time() - t0
print("kNN complexity and bandwidth selected and model fitted in %.6f s" % runtime)

train_score_mse = mean_squared_error(      sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_mae = mean_absolute_error(     sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_evs = explained_variance_score(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_me  = max_error(               sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))

test_score_mse = mean_squared_error(      sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_mae = mean_absolute_error(     sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_evs = explained_variance_score(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_me  = max_error(               sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_r2  = r2_score(                sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))

print("The model performance for testing set")
print("--------------------------------------")
print('MAE is {}'.format(test_score_mae))
print('MSE is {}'.format(test_score_mse))
print('EVS is {}'.format(test_score_evs))
print('ME is {}'.format(test_score_me))
print('R2 score is {}'.format(test_score_r2))

sorted_grid_params = sorted(gs.best_params_.items(), key=operator.itemgetter(0))

out_text = '\t'.join(['k-nearest-neighbour',
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

best_algorithm = gs.best_params_['algorithm']
best_n_neighbors = gs.best_params_['n_neighbors']
best_leaf_size = gs.best_params_['leaf_size']
best_weights = gs.best_params_['weights']
best_p = gs.best_params_['p']

outF = open("output.txt", "w")
print('best_algorithm = ', best_algorithm, file=outF)
print('best_n_neighbors = ', best_n_neighbors, file=outF)
print('best_leaf_size = ', best_leaf_size, file=outF)
print('best_weights = ', best_weights, file=outF)
print('best_p = ', best_p, file=outF)
print('R2 score is {}'.format(test_score_r2))
outF.close()

kn = KNeighborsRegressor(n_neighbors=best_n_neighbors,
                         algorithm=best_algorithm,
                         leaf_size=best_leaf_size,
                         weights=best_weights,
                         p=best_p)

t0 = time.time()
kn.fit(x_train, y_train.ravel())
kn_fit = time.time() - t0
print("kNN complexity and bandwidth selected and model fitted in %.6f s" % kn_fit)

t0 = time.time()
y_kn = kn.predict(x_test)
kn_predict = time.time() - t0
print("kNN prediction for %d inputs in %.6f s" % (x_test.shape[0], kn_predict))

# open a file to append
outF = open("output.txt", "a")
print("kNN complexity and bandwidth selected and model fitted in %.6f s" % kn_fit, file=outF)
print("kNN prediction for %d inputs in %.6f s" % (x_test.shape[0], kn_predict),file=outF)
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

plt.scatter(x_test_dim, y_test_dim, s=2, c='k', marker='o', label='Matlab')
plt.scatter(x_test_dim, y_kn_dim,   s=2, c='r', marker='+', label='k-Nearest Neighbour')
#plt.title(''Relaxation term $R_{ci}$ regression')
plt.ylabel('$k_{diss}$ $[m^3/s]$')
plt.xlabel('T [K] ')
plt.legend()
plt.tight_layout()
#plt.savefig("regression_kNN.eps", dpi=150, crop='false')
plt.savefig('../pdf/regression_kNN_'+dataset+'_'+Lev+'.pdf', dpi=150, crop='false')
#plt.show()
#plt.close('all')

# save the model to disk
dump(gs, '../model/model_kNN_'+dataset+'_'+Lev+'.sav')
