#!/usr/bin/env python
# coding: utf-8

import dask.array as da

import dask.dataframe as dd

from dask.diagnostics import ProgressBar
ProgressBar().register()

from dask.distributed import Client
client = Client()  # start distributed scheduler locally.  Launch dashboard

from joblib import parallel_backend

# Import database
import numpy as np
import pandas as pd

#%time df = dd.read_csv("/home/lk/Public/MLA/TransportCoeffsRegression/data/TCs_air5.csv").persist()
#%time dataset=np.loadtxt("/home/lk/Public/MLA/TransportCoeffsRegression/data/TCs_air5_MD2.txt")
#%time dataset = pd.read_csv("/home/lk/Public/MLA/TransportCoeffsRegression/data/TCs_air5_MD2.txt")
#x = dataset[:,0:7] # T, P, x_N2, x_O2, x_NO, x_N, x_O
#y = dataset[:,7:]  # D_cidk upper triangular matrix (Dij | j=>i)
#x = df[:,0:7] # T, P, x_N2, x_O2, x_NO, x_N, x_O
#y = df[:,7:]  # D_cidk upper triangular matrix (Dij | j=>i)
#dataset.head()

df.head(10)
#import h5py
#import xarray as xr
import os
import time
#filename = os.path.join('data', 'accounts.*.csv')
#filename
#target = os.path.join('data', 'accounts.h5')
#target
df_hdf = dd.read_hdf('myh4file.h5', ' ')
df_hdf.head()

#f = h5py.File(os.path.join('.', 'myh4file.h5'), mode='r')

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

from sklearn.inspection import permutation_importance

from joblib import dump, load
import pickle

from sklearn.tree import DecisionTreeRegressor

from sklearn.multioutput import MultiOutputRegressor

n_jobs = 1
trial  = 1

# The data is then split into training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=69)

sc_x = StandardScaler()
sc_y = StandardScaler()

sc_x.fit(x_train)
x_train = sc_x.fit_transform(x_train)
x_test  = sc_x.fit_transform(x_test)

sc_y.fit(y_train)
y_train = sc_y.transform(y_train)
y_test  = sc_y.transform(y_test)

dump(sc_x, open('../scaler/scaler_x_MD.pkl', 'wb'))
dump(sc_y, open('../scaler/scaler_y_MD.pkl', 'wb'))

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', x_test.shape)
print('Testing Labels Shape:', y_test.shape)

regr = DecisionTreeRegressor(criterion='mse',
                             splitter='best',
                             max_features='auto',
                             random_state=69)

regr = MultiOutputRegressor(estimator=regr)

t0 = time.time()
with parallel_backend("dask"):
    regr.fit(x_train, y_train)
regr_fit = time.time() - t0
print("Complexity and bandwidth selected and model fitted in %.6f s" % regr_fit)

t0 = time.time()
y_regr = regr.predict(x_test)
regr_predict = time.time() - t0
print("Prediction for %d inputs in %.6f s" % (x_test.shape[0], regr_predict))

x_test_dim = sc_x.inverse_transform(x_test)
y_test_dim = sc_y.inverse_transform(y_test)
y_regr_dim = sc_y.inverse_transform(y_regr)

plt.scatter(x_test_dim[:,0], y_test_dim[:,0], s=5, c='k', marker='o', label='KAPPA')
plt.scatter(x_test_dim[:,0], y_regr_dim[:,0], s=5, c='r', marker='d', label='k-Nearest Neighbour')
plt.scatter(x_test_dim[:,0], y_test_dim[:,1], s=5, c='k', marker='o', label='KAPPA')
plt.scatter(x_test_dim[:,0], y_regr_dim[:,1], s=5, c='r', marker='d', label='k-Nearest Neighbour')
#plt.scatter(x_test_dim[:,0], y_test_dim[:,2], s=5, c='k', marker='o', label='KAPPA')
#plt.scatter(x_test_dim[:,0], y_regr_dim[:,2], s=5, c='r', marker='d', label='k-Nearest Neighbour')
#plt.title('Shear viscosity regression with kNN')
#plt.ylabel(r'$\eta$ [Pa·s]')
plt.ylabel(' ')
plt.xlabel('T [K] ')
plt.legend()
plt.tight_layout()
plt.savefig("../pdf/regression_MD.pdf", dpi=150, crop='false')
plt.show()

# save the model to disk
dump(regr, '../model/model_MD.sav')
