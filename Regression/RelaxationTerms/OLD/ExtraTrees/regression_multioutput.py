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
from joblib import dump, load
import pickle
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

n_jobs = 1
trial  = 1

#dataset=np.loadtxt("../data/datarelax.txt")
#dataset=np.loadtxt("../data/datasetDR.txt")
#dataset=np.loadtxt("../data/datasetVT.txt")
dataset=np.loadtxt("../data/datasetVV.txt")

x = dataset[:,2:3]  # 0: x [m], 1: t [s], 2: T [K]
y = dataset[:,3:51] # Rci (relaxation source terms)

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

regr = ExtraTreesRegressor(n_estimators=1000,
                           min_weight_fraction_leaf=0.0,
                           max_depth=None,
                           max_leaf_nodes=100,
                           min_samples_split=10,
                           min_samples_leaf=1)
regr = MultiOutputRegressor(estimator=regr)

t0 = time.time()
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

plt.scatter(x_test_dim, y_test_dim[:,10], s=2, c='k', marker='o', label='Matlab')
plt.scatter(x_test_dim, y_regr_dim[:,10], s=2, c='r', marker='+', label='ExtraTrees, i=10')

plt.scatter(x_test_dim, y_test_dim[:,20], s=2, c='k', marker='o', label='Matlab')
plt.scatter(x_test_dim, y_regr_dim[:,20], s=2, c='g', marker='+', label='ExtraTrees, i=20')

plt.scatter(x_test_dim, y_test_dim[:,30], s=2, c='k', marker='o', label='Matlab')
plt.scatter(x_test_dim, y_regr_dim[:,30], s=2, c='b', marker='+', label='ExtraTrees, i=30')

plt.scatter(x_test_dim, y_test_dim[:,40], s=2, c='k', marker='o', label='Matlab')
plt.scatter(x_test_dim, y_regr_dim[:,40], s=2, c='m', marker='+', label='ExtraTrees, i=40')

#plt.title('Relaxation term $R_{ci}$ regression')
plt.ylabel('$R_{ci}$ $[J/m^3/s]$')
plt.xlabel('T [K]')
plt.legend()
plt.tight_layout()
#plt.savefig("regression_MO_ET_DR.eps", dpi=150, crop='false')
#plt.savefig("regression_MO_ET_DR.pdf", dpi=150, crop='false')
#plt.savefig("regression_MO_ET_VT.eps", dpi=150, crop='false')
#plt.savefig("regression_MO_ET_VT.pdf", dpi=150, crop='false')
plt.savefig("regression_MO_ET_VV.eps", dpi=150, crop='false')
plt.savefig("regression_MO_ET_VV.pdf", dpi=150, crop='false')
plt.show()

# save the model to disk
#dump(regr, 'model_MO_KR_DR.sav')
#dump(regr, 'model_MO_KR_VT.sav')
#dump(regr, 'model_MO_KR_VV.sav')
