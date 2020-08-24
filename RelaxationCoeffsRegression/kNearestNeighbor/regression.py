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
from sklearn import neighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.neighbors import NearestNeighbors
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

# kNN
hyper_params = [{'algorithm': ('ball_tree', 'kd_tree', 'brute',),
                 'n_neighbors': (1,2,3,4,5,6,7,8,9,10,),
                 'leaf_size': (1, 10, 20, 30, 100,),
                 'weights': ('uniform', 'distance',),
                 'p': (1,2,),}]

est=neighbors.KNeighborsRegressor()

gs = GridSearchCV(est, cv=10, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2')

t0 = time.time()
gs.fit(x_train, y_train.ravel())
runtime = time.time() - t0
print("kNN complexity and bandwidth selected and model fitted in %.3f s" % runtime)

train_score_mse = mean_squared_error(      sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_mae = mean_absolute_error(     sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_evs = explained_variance_score(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_me  = max_error(               sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))

test_score_mse = mean_squared_error(      sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_mae = mean_absolute_error(     sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_evs = explained_variance_score(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_me  = max_error(               sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))

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

plt.scatter(x_test_dim, y_test_dim, s=5, c='r', marker='o', label='Matlab')
plt.scatter(x_test_dim, y_kn_dim,   s=2, c='k', marker='d', label='k-Nearest Neighbour')
#plt.title(''Relaxation term $R_{ci}$ regression')
plt.ylabel('$R_{ci}$ $[J/m^3/s]$')
plt.xlabel('T [K] ')
plt.legend()
plt.tight_layout()
plt.savefig("regression_kNN.eps", dpi=150, crop='false')
plt.savefig("regression_kNN.pdf", dpi=150, crop='false')
plt.show()
