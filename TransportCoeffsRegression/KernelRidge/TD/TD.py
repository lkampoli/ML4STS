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

from sklearn import kernel_ridge
from sklearn.kernel_ridge import KernelRidge

from joblib import dump, load
import pickle

import pandas as pd

n_jobs = 1
trial  = 1

#dataset = pd.read_csv('../../data/dataset_TD.csv', skiprows=[i for i in range(1,215600,2)])
#dataset = np.asarray(dataset)
#print(dataset.shape)
#np.savetxt('../../data/tmp.csv', dataset, delimiter=',')
#dataset = pd.read_csv('../../data/tmp.csv', skiprows=[i for i in range(1,215600,2)])
#dataset = np.asarray(dataset)
#print(dataset.shape)
#np.savetxt('../../data/tmp.csv', dataset, delimiter=',')
#dataset = pd.read_csv('../../data/tmp.csv', skiprows=[i for i in range(1,215600,2)])
#dataset = np.asarray(dataset)
#print(dataset.shape)
#print(dataset[1000,:])

#dataset=np.loadtxt("../../data/dataset_TD.csv", delimiter=",")
dataset=np.loadtxt("../../data/dataset_TD_10000K.csv", delimiter=",")
x=dataset[:,1:3]
y=dataset[:,3] # 0: X, 1: T, 2: i, 3: TD 19597

# Plot dataset
#plt.scatter(x[:,1], dataset[:,2], s=0.5)
plt.scatter(x[:,0], y, s=0.5)
plt.title('Shear viscosity')
plt.xlabel('T [K]')
plt.ylabel(r'$\eta$')
plt.show()

#sys.exit()

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

# The data is then split into training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=69)

y_train = np.reshape(y_train, (-1,1))
y_test  = np.reshape(y_test, (-1,1))

#sc_x = MinMaxScaler(feature_range=(0, 1))
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

hyper_params = [{
                 #'kernel': ('poly','rbf',),
                 #'alpha': (1e-4,1e-2,0.1,1,10,),
                 #'gamma': (0.01,0.1,1,10,100,),
                 'kernel': ('rbf',),
                 'alpha': (1e-2,),
                 'gamma': (100,),
}]

est=kernel_ridge.KernelRidge()
gs = GridSearchCV(est, cv=2, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2')

t0 = time.time()
gs.fit(x_train, y_train.ravel())
runtime = time.time() - t0
print("KRR complexity and bandwidth selected and model fitted in %.6f s" % runtime)

train_score_mse  = mean_squared_error(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_mae  = mean_absolute_error(sc_y.inverse_transform(y_train),sc_y.inverse_transform(gs.predict(x_train)))
train_score_evs  = explained_variance_score(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_me   = max_error(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))

test_score_mse  = mean_squared_error(sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))
test_score_mae  = mean_absolute_error(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_evs  = explained_variance_score(sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))
test_score_me   = max_error(sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))
test_score_r2   = r2_score(                sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))

print("The model performance for testing set")
print("--------------------------------------")
print('MAE is {}'.format(test_score_mae))
print('MSE is {}'.format(test_score_mse))
print('EVS is {}'.format(test_score_evs))
print('ME is {}'.format(test_score_me))
print('R2 score is {}'.format(test_score_r2))

sorted_grid_params = sorted(gs.best_params_.items(), key=operator.itemgetter(0))

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
print("KR complexity and bandwidth selected and model fitted in %.6f s" % kr_fit)

t0 = time.time()
y_kr = kr.predict(x_test)
kr_predict = time.time() - t0
print("KR prediction for %d inputs in %.6f s" % (x_test.shape[0], kr_predict))

# open a file to append
outF = open("output.txt", "a")
print("KR complexity and bandwidth selected and model fitted in %.6f s" % kr_fit, file=outF)
print("KR prediction for %d inputs in %.6f s" % (x_test.shape[0], kr_predict),file=outF)
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
plt.scatter(x_test_dim[:,1], y_kr_dim[:],   s=2, c='green',   marker='p', label='Kernel Ridge')
#plot.title('Thermal diffusion regression with KR')
plt.ylabel(r'$D_T$ $[m^2/s]$')
plt.xlabel('T [K] ')
plt.legend()
plt.tight_layout()
plt.savefig("TD_KR.pdf", dpi=150, crop='false')
plt.show()

# save the model to disk
dump(gs, 'model_KR.sav')
