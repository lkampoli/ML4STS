#!/usr/bin/env python

import time
import sys
sys.path.insert(0, '../../Utilities/')
from plotting import newfig, savefig
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import operator
import itertools
from sklearn import metrics
from sklearn.metrics import *
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV #, learning_curve, cross_val_score
from sklearn import svm
from sklearn.svm import SVR
from joblib import dump, load
import pickle
from sklearn.multioutput import MultiOutputRegressor
#from sklearn.pipeline import Pipeline

n_jobs = 1
trial  = 1

dataset=np.loadtxt("../data/datarelax.txt")
print(dataset.shape)
x = dataset[:,0:1]   # Temperatures
y = dataset[:,1:50]  # Rci (relaxation source terms)

#for i in range (2,48):
#    plt.scatter(dataset[:,0:1], dataset[:,i], s=0.5, label=i)
##plt.title('$R_{ci}$ for $N_2/N$')
#plt.xlabel('T [K]')
#plt.ylabel('$R_{ci}$ $[J/m^3/s]$')
##plt.legend()
#plt.tight_layout()
#plt.savefig("relaxation_terms.pdf")
#plt.savefig("relaxation_terms.eps")
#plt.show()

#selected=[5, 10, 15, 20, 25, 30, 35, 40, 45]

##fig = plt.figure()
##for Y in range (2,48):
##    if Y in selected:
##        mylabel="i = %s"%(Y); mycolor='blue'
##    else:
##        mylabel=None; mycolor='white'
##    plt.scatter(dataset[:,0:1], dataset[:,Y], s=0.5, label=mylabel)
##plt.scatter(dataset[:,0:1], dataset[:,1], c='b', s=0.5, label='i=1')
#plt.scatter(dataset[:,0:1], dataset[:,2], c='k', s=0.5, label='i=2')
##plt.scatter(dataset[:,0:1], dataset[:,3], c='r', s=0.5, label='i=3')
#plt.scatter(dataset[:,0:1], dataset[:,4], c='m', s=0.5, label='i=4')
##plt.scatter(dataset[:,0:1], dataset[:,5], c='g', s=0.5, label='i=5')
#plt.scatter(dataset[:,0:1], dataset[:,6], c='y', s=0.5, label='i=6')
##plt.scatter(dataset[:,0:1], dataset[:,7], c='c', s=0.5, label='i=7')
#plt.scatter(dataset[:,0:1], dataset[:,8], c='b', s=0.5, label='i=8')
##plt.scatter(dataset[:,0:1], dataset[:,9], c='b', s=0.5, label='i=9')
#plt.scatter(dataset[:,0:1], dataset[:,10], c='r', s=0.5, label='i=10')
#plt.scatter(dataset[:,0:1], dataset[:,12], c='g', s=0.5, label='i=12')
##plt.scatter(dataset[:,0:1], dataset[:,20], c='b', s=0.5, label='i=20')
##plt.scatter(dataset[:,0:1], dataset[:,30], c='g', s=0.5, label='i=30')
##plt.scatter(dataset[:,0:1], dataset[:,40], c='m', s=0.5, label='i=40')
#plt.xlabel('T [K]')
#plt.ylabel('$R_{ci}$ $[J/m^3/s]$')
#plt.legend()
#plt.tight_layout()
#plt.savefig("relaxation_terms.pdf")
#plt.savefig("relaxation_terms.eps")
#plt.show()

#### end plotting ###

# Here, I learn one specific level of R_ci spanning all temperatures
#x=dataset[:,0:1]  # Temperatures
#y=dataset[:,1:50] # Rci (relaxation source terms)

# 2D Plot
#plt.scatter(x, y, s=0.5)
#plt.title('$R_{ci}$ for $N_2/N$ and i = 10')
#plt.xlabel('T [K]')
#plt.ylabel('$R_{ci}$ $[J/m^3/s]$')
#plt.tight_layout()
#plt.savefig("relaxation_source_terms.pdf")
#plt.show()

## 3D Plot
#fig = plt.figure()
#ax = Axes3D(fig)
#ax.scatter(x, y[:,0], y[:,1] s=0.5)
##ax.set_xlabel('molar fraction', fontsize=20, rotation=150)
#ax.set_ylabel('T [K]')
## disable auto rotation
#ax.zaxis.set_rotate_label(False)
##ax.set_zlabel(r'$D_ij [m^2/s]$', fontsize=30, rotation = 0)
#plt.show()

###
# https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/
# https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/

# prepare dataset with input and output scalers, can be none
def get_dataset(input_scaler, output_scaler):

    # generate dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.80, test_size=0.20, random_state=42)

    ## copy of datasets
    #X_train_stand = x_train.copy()
    #X_test_stand  = x_test.copy()

    ## numerical features
    #num_cols = 49

    ## apply standardization on numerical features
    #for i in num_cols:

    ## fit on training data column
    #scale = StandardScaler().fit(X_train_stand[[i]])

    ## transform the training data column
    #X_train_stand[i] = scale.transform(X_train_stand[[i]])

    ## transform the testing data column
    #X_test_stand[i] = scale.transform(X_test_stand[[i]])

    # scale inputs
    if input_scaler is not None:

        #for i in [0, x_train.shape[1]]:
        for i in range(x_train.shape[1]):

            # fit scaler
            sc_x = input_scaler

            # fit on training data column
            scale = sc_x.fit(x_train[[i]])

            # transform the training data column
            x_train[i] = scale.transform(x_train[[i]])

            # transform the testing data column
            x_test[i] = scale.transform(x_test[[i]])

    if output_scaler is not None:

        print(y_train.shape[1])
        for i in [0, y_train.shape[1]]:

            # fit scaler
            sc_y = output_scaler

            # fit on training data column
            scale = sc_y.fit(y_train[[i]])

            # transform the training data column
            y_train[i] = scale.transform(y_train[[i]])

            # transform the testing data column
            y_test[i] = scale.transform(y_test[[i]])

    return x_train, y_train, x_test, y_test, sc_x, sc_y

###

#y=np.reshape(y, (-1,1))
#sc_x = StandardScaler()
#sc_y = StandardScaler()
#X = sc_x.fit_transform(x)
#Y = sc_y.fit_transform(y)

#x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.75, test_size=0.25, random_state=42)
#x_train_sc, x_test_sc, y_train_sc, y_test_sc = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=42)
#x_train_sc, x_test_sc, y_train_sc, y_test_sc = train_test_split(x, y, train_size=0.80, test_size=0.20, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.80, test_size=0.20, random_state=42)
#x_train_sc, x_test_sc, y_train_sc, y_test_sc = train_test_split(x, y, train_size=0.90, test_size=0.10, random_state=42)
#x_train_sc, x_test_sc, y_train_sc, y_test_sc = train_test_split(x, y, train_size=0.50, test_size=0.50, random_state=42)

print("x_train.shape, y_train.shape")
print(x_train.shape, y_train.shape)

# copy of datasets
X_train_stand = x_train.copy()
X_test_stand  = x_test.copy()
Y_train_stand = y_train.copy()
Y_test_stand  = y_test.copy()

print("X_train_stand.shape, Y_train_stand.shape")
print(X_train_stand.shape, Y_train_stand.shape)

# numerical features
num_cols = 48

regr = SVR(kernel='rbf', epsilon=0.015, C=1000., gamma='auto', coef0=0.0)
#regr = SVR()
#regr = MultiOutputRegressor(estimator=regr)

scale = StandardScaler().fit(X_train_stand)
X_train_stand = scale.transform(X_train_stand)
X_test_stand = scale.transform(X_test_stand)

print(y_train.shape)

# apply standardization on numerical features
#for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]: #range(num_cols):
for i in range(num_cols):

    # fit on training data column
    scale = StandardScaler().fit(Y_train_stand[[i]])

    # transform the training data column
    Y_train_stand[i] = scale.transform(Y_train_stand[[i]])

    # transform the testing data column
    Y_test_stand[i] = scale.transform(Y_test_stand[[i]])

    # train
    t0 = time.time()
    regr.fit(X_train_stand, Y_train_stand[:,i])
    regr_fit = time.time() - t0
    print("Complexity and bandwidth selected and model fitted in %.6f s" % regr_fit)

    # predict
    t0 = time.time()
    y_regr = regr.predict(X_test_stand)
    regr_predict = time.time() - t0
    print("Prediction for %d inputs in %.6f s" % (x_test.shape[0], regr_predict))

    #print(y_regr.shape)
    #print(Y_test_stand.shape)

    print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(Y_test_stand[:,i], y_regr))
    print('Mean Squared Error (MSE):', metrics.mean_squared_error(Y_test_stand[:,i], y_regr))
    print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(Y_test_stand[:,i], y_regr)))

    #x_test_dim = sc_x.inverse_transform(x_test)
    #y_test_dim = sc_y.inverse_transform(y_test)
    #y_regr_dim = sc_y.inverse_transform(y_regr)

    #x_test_dim[i] = sc_x.inverse_transform(x_test[i])
    #y_test_dim[i] = sc_y.inverse_transform(y_test[i])
    #y_regr_dim[i] = sc_y.inverse_transform(y_regr[i])

#sc_x = StandardScaler()
#sc_y = StandardScaler()
#sc_x = MinMaxScaler()
#sc_y = MinMaxScaler()
#sc_x = MaxAbsScaler()
#sc_y = MaxAbsScaler()
#sc_x = RobustScaler()
#sc_y = RobustScaler()

#x_train = sc_x.fit_transform(x_train_sc)
#y_train = sc_y.fit_transform(y_train_sc)
#x_test  = sc_x.fit_transform(x_test_sc)
#y_test  = sc_y.fit_transform(y_test_sc)

#x_train, y_train, x_test, y_test, sc_x, sc_y = get_dataset(StandardScaler(), StandardScaler())
#x_train, y_train, x_test, y_test, sc_x, sc_y = get_dataset(MinMaxScaler(), StandardScaler())

#print('Training Features Shape:', x_train.shape)
#print('Training Labels Shape:',   y_train.shape)
#print('Testing Features Shape:',  x_test.shape)
#print('Testing Labels Shape:',    y_test.shape)

#regr = SVR(kernel='rbf', epsilon=0.015, C=1000., gamma='auto', coef0=0.0)
#regr = SVR()
#regr = MultiOutputRegressor(estimator=regr)
#print(regr)

#t0 = time.time()
#regr.fit(x_train, y_train)
#regr_fit = time.time() - t0
#print("Complexity and bandwidth selected and model fitted in %.6f s" % regr_fit)

#t0 = time.time()
#y_regr = regr.predict(x_test)
#regr_predict = time.time() - t0
#print("Prediction for %d inputs in %.6f s" % (x_test.shape[0], regr_predict))

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

#x_test_dim = sc_x.inverse_transform(x_test)
#y_test_dim = sc_y.inverse_transform(y_test)
#y_regr_dim = sc_y.inverse_transform(y_regr)

#for i in [0, x_test.shape[1]]:
#    x_test_dim[i] = sc_x.inverse_transform(x_test[i])
#
#for i in [0, y_test.shape[1]]:
#    y_test_dim[i] = sc_y.inverse_transform(y_test[i])
#
#for i in [0, y_regr.shape[1]]:
#    y_regr_dim[i] = sc_y.inverse_transform(y_regr[i])

#plt.scatter(x_test_dim, y_test_dim[:,2], s=5, c='k', marker='o', label='Matlab i = 2')
#plt.scatter(x_test_dim, y_regr_dim[:,2], s=2, c='r', marker='d', label='SVR i = 2')
#
#plt.scatter(x_test_dim, y_test_dim[:,4], s=5, c='k', marker='o', label='Matlab i = 4')
#plt.scatter(x_test_dim, y_regr_dim[:,4], s=2, c='b', marker='d', label='SVR i = 4')
#
#plt.scatter(x_test_dim, y_test_dim[:,6], s=5, c='k', marker='o', label='Matlab i = 6')
#plt.scatter(x_test_dim, y_regr_dim[:,6], s=2, c='g', marker='d', label='SVR i = 6')
#
#plt.scatter(x_test_dim, y_test_dim[:,8], s=5, c='k', marker='o', label='Matlab i = 8')
#plt.scatter(x_test_dim, y_regr_dim[:,8], s=2, c='m', marker='d', label='SVR i = 8')
#
#plt.scatter(x_test_dim, y_test_dim[:,10], s=5, c='k', marker='o', label='Matlab i = 10')
#plt.scatter(x_test_dim, y_regr_dim[:,10], s=2, c='y', marker='d', label='SVR i = 10')
#
#plt.scatter(x_test_dim, y_test_dim[:,12], s=5, c='k', marker='o', label='Matlab i = 12')
#plt.scatter(x_test_dim, y_regr_dim[:,12], s=2, c='c', marker='d', label='SVR i = 12')

#plt.scatter(x_test_dim, y_test_dim[:,10], s=5, c='k', marker='o', label='Matlab i = 10')
#plt.scatter(x_test_dim, y_regr_dim[:,10], s=2, c='r', marker='d', label='SVR i = 10')

#plt.scatter(x_test_dim, y_test_dim[:,15], s=5, c='k', marker='o', label='Matlab i = 15')
#plt.scatter(x_test_dim, y_regr_dim[:,15], s=2, c='g', marker='d', label='SVR i = 15')

#plt.scatter(x_test_dim, y_test_dim[:,20], s=5, c='k', marker='o', label='Matlab i = 20')
#plt.scatter(x_test_dim, y_regr_dim[:,20], s=2, c='b', marker='d', label='SVR i = 20')

#plt.scatter(x_test_dim, y_test_dim[:,25], s=5, c='k', marker='o', label='Matlab i = 25')
#plt.scatter(x_test_dim, y_regr_dim[:,25], s=2, c='y', marker='d', label='SVR i = 25')

#plt.scatter(x_test_dim, y_test_dim[:,30], s=5, c='k', marker='o', label='Matlab i = 30')
#plt.scatter(x_test_dim, y_regr_dim[:,30], s=2, c='m', marker='d', label='SVR i = 30')
#
#plt.scatter(x_test_dim, y_test_dim[:,35], s=5, c='k', marker='o', label='Matlab i = 35')
#plt.scatter(x_test_dim, y_regr_dim[:,35], s=2, c='c', marker='d', label='SVR i = 35')
#
#plt.scatter(x_test_dim, y_test_dim[:,40], s=5, c='k', marker='o', label='Matlab i = 40')
#plt.scatter(x_test_dim, y_regr_dim[:,40], s=2, c='r', marker='d', label='SVR i = 40')
#
#plt.scatter(x_test_dim, y_test_dim[:,45], s=5, c='k', marker='o', label='Matlab i = 45')
#plt.scatter(x_test_dim, y_regr_dim[:,45], s=2, c='g', marker='d', label='SVR i = 45')

#plt.title('Relaxation term $R_{ci}$ regression')
#plt.ylabel('$R_{ci}$ $[J/m^3/s]$')
#plt.xlabel('T [K] ')
#plt.legend()
#plt.tight_layout()
#plt.savefig("regression_MO_SVR.eps", dpi=150, crop='false')
#plt.savefig("regression_MO_SVR.pdf", dpi=150, crop='false')
#plt.show()
