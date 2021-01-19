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
#from sklearn import ensemble
#from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load
import pickle
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn import svm
from sklearn.svm import SVR

n_jobs = -1
trial  = 1

dataset_DR=np.loadtxt("../data/datasetDR.txt")
dataset_VT=np.loadtxt("../data/datasetVT.txt")
dataset_VV=np.loadtxt("../data/datasetVV.txt")

x_DR = dataset_DR[:,2:3]  # 0: x [m], 1: t [s], 2: T [K]
y_DR = dataset_DR[:,3:51] # Rci (relaxation source terms)
x_VT = dataset_VT[:,2:3]  # 0: x [m], 1: t [s], 2: T [K]
y_VT = dataset_VT[:,3:51] # Rci (relaxation source terms)
x_VV = dataset_VV[:,2:3]  # 0: x [m], 1: t [s], 2: T [K]
y_VV = dataset_VV[:,3:51] # Rci (relaxation source terms)

x_DR_train, x_DR_test, y_DR_train, y_DR_test = train_test_split(x_DR, y_DR, train_size=0.75, test_size=0.25, random_state=69)
x_VT_train, x_VT_test, y_VT_train, y_VT_test = train_test_split(x_VT, y_VT, train_size=0.75, test_size=0.25, random_state=69)
x_VV_train, x_VV_test, y_VV_train, y_VV_test = train_test_split(x_VV, y_VV, train_size=0.75, test_size=0.25, random_state=69)

sc_x_DR = StandardScaler()
sc_y_DR = StandardScaler()
sc_x_VT = StandardScaler()
sc_y_VT = StandardScaler()
sc_x_VV = StandardScaler()
sc_y_VV = StandardScaler()

# fit scaler
sc_x_DR.fit(x_DR_train)
sc_x_VT.fit(x_VT_train)
sc_x_VV.fit(x_VV_train)
# transform training datasetx
x_DR_train = sc_x_DR.transform(x_DR_train)
x_VT_train = sc_x_VT.transform(x_VT_train)
x_VV_train = sc_x_VV.transform(x_VV_train)
# transform test dataset
x_DR_test = sc_x_DR.transform(x_DR_test)
x_VT_test = sc_x_VT.transform(x_VT_test)
x_VV_test = sc_x_VV.transform(x_VV_test)

# fit scaler on training dataset
sc_y_DR.fit(y_DR_train)
sc_y_VT.fit(y_VT_train)
sc_y_VV.fit(y_VV_train)
# transform training dataset
y_DR_train = sc_y_DR.transform(y_DR_train)
y_VT_train = sc_y_VT.transform(y_VT_train)
y_VV_train = sc_y_VV.transform(y_VV_train)
# transform test dataset
y_DR_test = sc_y_DR.transform(y_DR_test)
y_VT_test = sc_y_VT.transform(y_VT_test)
y_VV_test = sc_y_VV.transform(y_VV_test)

regr = SVR(kernel='rbf',
           gamma='scale',
           C=100.,
           epsilon=0.01,
           coef0=0.0)
regr = MultiOutputRegressor(estimator=regr)

regr.fit(x_DR_train, y_DR_train)
y_DR_regr = regr.predict(x_DR_test)
regr.fit(x_VT_train, y_VT_train)
y_VT_regr = regr.predict(x_VT_test)
regr.fit(x_VV_train, y_VV_train)
y_VV_regr = regr.predict(x_VV_test)

# open a file to append
#outF = open("output_MO.txt", "a")
#print("Complexity and bandwidth selected and model fitted in %.6f s" % regr_fit, file=outF)
#print("Prediction for %d inputs in %.6f s" % (x_test.shape[0], regr_predict),file=outF)
#print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_regr), file=outF)
#print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_regr), file=outF)
#print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_regr)), file=outF)
#outF.close()

#print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_regr))
#print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_regr))
#print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_regr)))

x_DR_test_dim = sc_x_DR.inverse_transform(x_DR_test)
y_DR_test_dim = sc_y_DR.inverse_transform(y_DR_test)
y_DR_regr_dim = sc_y_DR.inverse_transform(y_DR_regr)
x_VT_test_dim = sc_x_VT.inverse_transform(x_VT_test)
y_VT_test_dim = sc_y_VT.inverse_transform(y_VT_test)
y_VT_regr_dim = sc_y_VT.inverse_transform(y_VT_regr)
x_VV_test_dim = sc_x_VV.inverse_transform(x_VV_test)
y_VV_test_dim = sc_y_VV.inverse_transform(y_VV_test)
y_VV_regr_dim = sc_y_VV.inverse_transform(y_VV_regr)

print(y_DR_test_dim.shape, y_DR_regr_dim.shape)
print(y_VT_test_dim.shape, y_VT_regr_dim.shape)
print(y_VV_test_dim.shape, y_VV_regr_dim.shape)

plt.scatter(x_DR_test_dim, y_DR_test_dim[:,10], s=8, c='k', marker='o', label='Matlab')
plt.scatter(x_DR_test_dim, y_DR_regr_dim[:,10], s=8, c='r', marker='+', label='SupportVectorMachine, DR, i=10')

plt.scatter(x_VT_test_dim, y_VT_test_dim[:,10], s=8, c='k', marker='o', label='Matlab')
plt.scatter(x_VT_test_dim, y_VT_regr_dim[:,10], s=8, c='g', marker='+', label='SupportVectorMachine, VT, i=10')

plt.scatter(x_VV_test_dim, y_VV_test_dim[:,10], s=8, c='k', marker='o', label='Matlab')
plt.scatter(x_VV_test_dim, y_VV_regr_dim[:,10], s=8, c='b', marker='+', label='SupportVectorMachine, VV, i=10')

plt.scatter(x_DR_test_dim, y_DR_test_dim[:,20], s=8, c='k', marker='o', label='Matlab')
plt.scatter(x_DR_test_dim, y_DR_regr_dim[:,20], s=8, c='r', marker='*', label='SupportVectorMachine, DR, i=20')

plt.scatter(x_VT_test_dim, y_VT_test_dim[:,20], s=8, c='k', marker='o', label='Matlab')
plt.scatter(x_VT_test_dim, y_VT_regr_dim[:,20], s=8, c='g', marker='*', label='SupportVectorMachine, VT, i=20')

plt.scatter(x_VV_test_dim, y_VV_test_dim[:,20], s=8, c='k', marker='o', label='Matlab')
plt.scatter(x_VV_test_dim, y_VV_regr_dim[:,20], s=8, c='b', marker='*', label='SupportVectorMachine, VV, i=20')

plt.scatter(x_DR_test_dim, y_DR_test_dim[:,30], s=8, c='k', marker='o', label='Matlab')
plt.scatter(x_DR_test_dim, y_DR_regr_dim[:,30], s=8, c='r', marker='^', label='SupportVectorMachine, DR, i=30')

plt.scatter(x_VT_test_dim, y_VT_test_dim[:,30], s=8, c='k', marker='o', label='Matlab')
plt.scatter(x_VT_test_dim, y_VT_regr_dim[:,30], s=8, c='g', marker='^', label='SupportVectorMachine, VT, i=30')

plt.scatter(x_VV_test_dim, y_VV_test_dim[:,30], s=8, c='k', marker='o', label='Matlab')
plt.scatter(x_VV_test_dim, y_VV_regr_dim[:,30], s=8, c='b', marker='^', label='SupportVectorMachine, VV, i=30')

plt.scatter(x_DR_test_dim, y_DR_test_dim[:,40], s=8, c='k', marker='o', label='Matlab')
plt.scatter(x_DR_test_dim, y_DR_regr_dim[:,40], s=8, c='r', marker='X', label='SupportVectorMachine, DR, i=40')

plt.scatter(x_VT_test_dim, y_VT_test_dim[:,40], s=8, c='k', marker='o', label='Matlab')
plt.scatter(x_VT_test_dim, y_VT_regr_dim[:,40], s=8, c='g', marker='X', label='SupportVectorMachine, VT, i=40')

plt.scatter(x_VV_test_dim, y_VV_test_dim[:,40], s=8, c='k', marker='o', label='Matlab')
plt.scatter(x_VV_test_dim, y_VV_regr_dim[:,40], s=8, c='b', marker='X', label='SupportVectorMachine, VV, i=40')

#plt.title('Relaxation term $R_{ci}$ regression')
plt.ylabel('$R_{ci}$ $[J/m^3/s]$')
plt.xlabel('T [K]')
plt.legend(markerscale=1.5)
plt.tight_layout()
#plt.savefig("regression_MO_RF_DR_VT_VV.eps", dpi=150, crop='false')
plt.savefig("regression_MO_SVR_DR_VT_VV.pdf", dpi=150, crop='false')
plt.show()

# save the model to disk
#dump(regr, 'model_MO_RF_DR_VT_VV.sav')
