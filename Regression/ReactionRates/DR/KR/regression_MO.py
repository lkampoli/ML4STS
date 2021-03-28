#!/usr/bin/env python
# coding: utf-8

import time
import sys
import os
import shutil

sys.path.insert(0, '../../../../Utilities/')

from plotting import newfig, savefig
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import pandas as pd
import seaborn as sns

import operator
import itertools

from sklearn import metrics
from sklearn.metrics import *

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV

import joblib
from joblib import dump, load
import pickle

from sklearn.inspection import permutation_importance

from sklearn.tree import DecisionTreeRegressor
from sklearn import kernel_ridge
from sklearn.kernel_ridge import KernelRidge

n_jobs = 2

# Read database filename from command-line input argument
dataset = sys.argv[1]
folder  = dataset[9:14]
process = dataset[15:18]

print(dataset)
print(folder)
print(process)

# Parent Directory path
parent_dir = "./"+dataset

# Directories
models  = "models"
scalers = "scalers"
figures = "figures"

shutil.rmtree(dataset, ignore_errors=True) 
shutil.rmtree(models,  ignore_errors=True) 
shutil.rmtree(scalers, ignore_errors=True) 
shutil.rmtree(figures, ignore_errors=True) 

# Path
model  = os.path.join(parent_dir, models)
scaler = os.path.join(parent_dir, scalers)
figure = os.path.join(parent_dir, figures)

from pathlib import Path
Path("./"+dataset).mkdir(parents=True, exist_ok=True)

os.mkdir(model)
os.mkdir(scaler)
os.mkdir(figure)

print("Directory '%s' created" %models)
print("Directory '%s' created" %scalers)
print("Directory '%s' created" %figures)

# Import database
dataset_T = np.loadtxt("../data/Temperatures.csv")
dataset_k = np.loadtxt("../data/"+folder+"/"+process+"/"+dataset+".csv")

x = dataset_T.reshape(-1,1)
y = dataset_k[:,:]

print(dataset_T.shape)
print(dataset_k.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=69)

sc_x = StandardScaler()
sc_y = StandardScaler()

sc_x.fit(x_train)
x_train = sc_x.transform(x_train)
x_test  = sc_x.transform(x_test)

sc_y.fit(y_train)
y_train = sc_y.transform(y_train)
y_test  = sc_y.transform(y_test)

dump(sc_x, open(dataset+'/scalers/scaler_x_MO_'+dataset+'.pkl', 'wb'))
dump(sc_y, open(dataset+'/scalers/scaler_y_MO_'+dataset+'.pkl', 'wb'))

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:',   y_train.shape)
print('Testing Features Shape:',  x_test.shape)
print('Testing Labels Shape:',    y_test.shape)

# Kernel Ridge estimator
# https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html
hyper_params = [{'kernel': ('poly', 'rbf',),
                 'alpha': (1e-3, 1e-2, 1e-1, 0.0, 0.5, 1.,),
#                'degree': (),
#                'coef0': (),
                 'gamma': (0.1, 1, 2,),}]

est=kernel_ridge.KernelRidge()

# Exhaustive search over specified parameter values for the estimator
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
gs = GridSearchCV(est, cv=10, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2',
                  refit=True, pre_dispatch='n_jobs', error_score=np.nan, return_train_score=True)

t0 = time.time()
gs.fit(x_train, y_train)
runtime = time.time() - t0
print("Training time: %.6f s" % runtime)

train_score_mse  = mean_squared_error(      sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_mae  = mean_absolute_error(     sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_evs  = explained_variance_score(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
#train_score_me   = max_error(               sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
#train_score_msle = mean_squared_log_error(  sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_r2   = r2_score(                sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))

test_score_mse  = mean_squared_error(      sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_mae  = mean_absolute_error(     sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_evs  = explained_variance_score(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
#test_score_me   = max_error(               sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
#test_score_msle = mean_squared_log_error(  sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_r2   = r2_score(                sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))

print()
print("The model performance for training set")
print("--------------------------------------")
print('MAE is      {}'.format(train_score_mae ))
print('MSE is      {}'.format(train_score_mse ))
print('EVS is      {}'.format(train_score_evs ))
#print('ME is       {}'.format(train_score_me  ))
#print('MSLE is     {}'.format(train_score_msle))
print('R2 score is {}'.format(train_score_r2  ))
print()
print("The model performance for testing set" )
print("--------------------------------------")
print('MAE is      {}'.format(test_score_mae ))
print('MSE is      {}'.format(test_score_mse ))
print('EVS is      {}'.format(test_score_evs ))
#print('ME is       {}'.format(test_score_me  ))
#print('MSLE is     {}'.format(test_score_msle))
print('R2 score is {}'.format(test_score_r2  ))
print()
print("Best parameters set found on development set:")
print(gs.best_params_)
print()

t0 = time.time()
y_regr = regr.predict(x_test)
regr_predict = time.time() - t0
print("Prediction for %d inputs in %.6f s" % (x_test.shape[0], regr_predict))

with open(dataset+'/output.log', 'w') as f:
    print("Training time: %.6f s"   % runtime,      file=f)
    print("Prediction time: %.6f s" % regr_predict, file=f)
    print(" ",                                      file=f)
    print("The model performance for training set", file=f)
    print("--------------------------------------", file=f)
    print('MAE is      {}'.format(train_score_mae), file=f)
    print('MSE is      {}'.format(train_score_mse), file=f)
    print('EVS is      {}'.format(train_score_evs), file=f)
#    print('ME is       {}'.format(train_score_me),  file=f)
#    print('MSLE is     {}'.format(train_score_msle),file=f)
    print('R2 score is {}'.format(train_score_r2),  file=f)
    print(" ",                                      file=f)
    print("The model performance for testing set",  file=f)
    print("--------------------------------------", file=f)
    print('MAE is      {}'.format(test_score_mae),  file=f)
    print('MSE is      {}'.format(test_score_mse),  file=f)
    print('EVS is      {}'.format(test_score_evs),  file=f)
#    print('ME is       {}'.format(test_score_me),   file=f)
#    print('MSLE is     {}'.format(test_score_msle), file=f)
    print('R2 score is {}'.format(test_score_r2),   file=f)
    print(" ",                                      file=f)
    print("Best parameters set found on dev set:",  file=f)
    print(gs.best_params_,                          file=f)

x_test_dim = sc_x.inverse_transform(x_test)
y_test_dim = sc_y.inverse_transform(y_test)
y_regr_dim = sc_y.inverse_transform(y_regr)

plt.scatter(x_test_dim, y_test_dim[:,5], s=2, c='k', marker='o', label='Matlab')
plt.scatter(x_test_dim, y_regr_dim[:,5], s=2, c='purple', marker='+', label='DT, i=5')

plt.scatter(x_test_dim, y_test_dim[:,10], s=2, c='k', marker='o', label='Matlab')
plt.scatter(x_test_dim, y_regr_dim[:,10], s=2, c='r', marker='+', label='DT, i=10')

plt.scatter(x_test_dim, y_test_dim[:,15], s=2, c='k', marker='o', label='Matlab')
plt.scatter(x_test_dim, y_regr_dim[:,15], s=2, c='c', marker='+', label='DT, i=15')

plt.scatter(x_test_dim, y_test_dim[:,20], s=2, c='k', marker='o', label='Matlab')
plt.scatter(x_test_dim, y_regr_dim[:,20], s=2, c='g', marker='+', label='DT, i=20')

plt.scatter(x_test_dim, y_test_dim[:,25], s=2, c='k', marker='o', label='Matlab')
plt.scatter(x_test_dim, y_regr_dim[:,25], s=2, c='y', marker='+', label='DT, i=25')

plt.scatter(x_test_dim, y_test_dim[:,30], s=2, c='k', marker='o', label='Matlab')
plt.scatter(x_test_dim, y_regr_dim[:,30], s=2, c='b', marker='+', label='DT, i=30')

plt.scatter(x_test_dim, y_test_dim[:,35], s=2, c='k', marker='o', label='Matlab')
plt.scatter(x_test_dim, y_regr_dim[:,35], s=2, c='m', marker='+', label='DT, i=35')

#plt.ylabel(r'$\eta$ [Pa·s]')
plt.xlabel('T [K] ')
plt.legend()
plt.tight_layout()
plt.savefig(dataset+'/figures/regression_MO_'+dataset+'.pdf')
#plt.show()
plt.close()

# save the model to disk
dump(gs, dataset+'/models/model_MO_'+dataset+'.sav')
