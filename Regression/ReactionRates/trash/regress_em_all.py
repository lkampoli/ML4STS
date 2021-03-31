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

from joblib import dump, load
import pickle

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn import ensemble
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn import neighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn import kernel_ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn import svm
from sklearn.svm import SVR

import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

#from IPython.display import clear_output
#from livelossplot import PlotLossesKeras
#from keras.callbacks import TensorBoard
from keras.utils.vis_utils import plot_model
from keras.models import load_model
#from ann_visualizer.visualize import ann_viz;
from keras.models import model_from_json
from keras_sequential_ascii import keras2ascii
#from livelossplot import PlotLossesKeras
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras import regularizers

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


def est_DT(est):
   hyper_params = [{'criterion': ('mse', 'friedman_mse', 'mae'), 
                    'splitter': ('best', 'random'),             
                    'max_features': ('auto', 'sqrt', 'log2'),  
   }]
   
   est = DecisionTreeRegressor()


def est_ET(est):
   hyper_params = [{'n_estimators': (1, 100,),
                    'min_weight_fraction_leaf': (0.0, 0.25, 0.5,),
                    'max_features': ('sqrt','log2','auto', None,),
                    'max_samples': (1,10,100,1000,),
                    'bootstrap': (True, False,),
                    'oob_score': (True, False,),
                    'warm_start': (True, False,),
                    'criterion': ('mse', 'mae',),
                    'max_depth': (1,10,100,None,),
                    'max_leaf_nodes': (2, 100,),
                    'min_samples_split': (10,),
                    'min_samples_leaf': (1,10,100,),
   }]
   
   est = ensemble.ExtraTreesRegressor()
   #regr = MultiOutputRegressor(estimator=est)


def est_GB(est):
   hyper_params = [{'n_estimators': (10, 100, 1000,),
                    'min_weight_fraction_leaf': (0.0, 0.1, 0.2, 0.3,),
                    'max_features': ('sqrt', 'log2', 'auto',),
                    'warm_start': (False, True),
                    'criterion': ('friedman_mse', 'mse', 'mae',),
                    'max_depth': (1, 10, 100, None,),
                    'min_samples_split': (2, 5, 10,),
                    'min_samples_leaf': (2, 5, 10,),
                    'loss': ('ls', 'lad', 'huber', 'quantile',),
   }]

   est = ensemble.GradientBoostingRegressor()


def est_HGB(est):
   hyper_params = [{'warm_start': (False, True),
                    'max_depth': (1, 10, 100, None,),
                    'min_samples_leaf': (2, 5, 10,),
                    'loss': ('ls', 'lad', 'huber', 'quantile',),
                    'max_leaf_nodes': (2, 10, 20, 30, 40, 50, 100,),
   }]

   est=ensemble.HistGradientBoostingRegressor()

def est_KN(est):
   hyper_params = [{'algorithm': ('ball_tree', 'kd_tree', 'brute', 'auto',),
                    'n_neighbors': (1, 5, 10, 20),
                    'leaf_size': (1, 10, 50, 100,),
                    'weights': ('uniform', 'distance',),
#                   'metric': ('minkowski', ),  
#                   'metric_params': (), 
                    'p': (1, 2,),}]

   est=neighbors.KNeighborsRegressor()


def est_KR(est):
   hyper_params = [{'kernel': ('poly', 'rbf',),
                    'alpha': (1e-3, 1e-2, 1e-1, 0.0, 0.5, 1.,),
#                   'degree': (),
#                   'coef0': (),
                    'gamma': (0.1, 1, 2,),}]

   est=kernel_ridge.KernelRidge()


def est_MLP(est):
   hyper_params = [{'hidden_layer_sizes': (10, 50, 100, 150, 200,),
                    'activation' : ('tanh', 'relu',),
                    'solver' : ('lbfgs','adam','sgd',), 
                    'learning_rate' : ('constant', 'invscaling', 'adaptive',),
                    'nesterovs_momentum': (True, False,),
                    'alpha': (0.00001, 0.0001, 0.001, 0.01, 0.1, 0.0,),
                    'warm_start': (True, False,),
                    'early_stopping': (True, False,),
                    'max_iter': (1000,)
   }]

   est = MLPRegressor()

def est_RF(est):
   hyper_params = [{'n_estimators': (1, 50, 100,),
                    'min_weight_fraction_leaf': (0.0, 0.25, 0.5,),
                    'max_features': ('sqrt', 'log2', 'auto',),
#                   'bootstrap': (True, False,),
#                   'oob_score': (True, False,),
#                   'warm_start': (True, False,),
                    'criterion': ('mse', 'mae',),
                    'max_depth': (1, 10, 100,),
                    'max_leaf_nodes': (2, 100,),
                    'min_samples_split': (2, 5, 10,),
                    'min_impurity_decrease': (0.1, 0.2, 0.3,),
                    'min_samples_leaf': (1, 10, 100,),
#                   'min_impurity_split':= (), 
#                   'ccp_alpha': (),
#                   'max_samples': (),
   }]

   est=ensemble.RandomForestRegressor(random_state=69)


def est_SVM(est, hyper_params):
   hyper_params = [{'kernel': ('poly', 'rbf',),
                    'gamma': ('scale', 'auto',),
                    'C': (1e-1, 1e0, 1e1,),
                    'epsilon': (1e-2, 1e-1, 1e0, 1e1,),
                    'coef0': (0.0, 0.1, 0.2,),
   }]
  
   est = svm.SVR()


def GridSearch(est, hyper_params, gs, n_jobs):
   gs = GridSearchCV(est, cv=10, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2',
                     refit=True, pre_dispatch='n_jobs', error_score=np.nan, return_train_score=True)

def train(gs):
   t0 = time.time()
   gs.fit(x_train, y_train)
   runtime = time.time() - t0
   print("Training time: %.6f s" % runtime)


def predict(gs, x_test, y_regr):
   t0 = time.time()
   y_regr = gs.predict(x_test)
   regr_predict = time.time() - t0
   print("Prediction for %d inputs in %.6f s" % (x_test.shape[0], regr_predict))


def print_scores(gs, sc_x, sc_y, x_train, y_train, x_test, y_test, dataset):
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

   with open(dataset+'/output.log', 'w') as f:
       print("Training time: %.6f s"   % runtime,      file=f)
       print("Prediction time: %.6f s" % regr_predict, file=f)
       print(" ",                                      file=f)
       print("The model performance for training set", file=f)
       print("--------------------------------------", file=f)
       print('MAE is      {}'.format(train_score_mae), file=f)
       print('MSE is      {}'.format(train_score_mse), file=f)
       print('EVS is      {}'.format(train_score_evs), file=f)
       #print('ME is       {}'.format(train_score_me),  file=f)
       #print('MSLE is     {}'.format(train_score_msle),file=f)
       print('R2 score is {}'.format(train_score_r2),  file=f)
       print(" ",                                      file=f)
       print("The model performance for testing set",  file=f)
       print("--------------------------------------", file=f)
       print('MAE is      {}'.format(test_score_mae),  file=f)
       print('MSE is      {}'.format(test_score_mse),  file=f)
       print('EVS is      {}'.format(test_score_evs),  file=f)
       #print('ME is       {}'.format(test_score_me),   file=f)
       #print('MSLE is     {}'.format(test_score_msle), file=f)
       print('R2 score is {}'.format(test_score_r2),   file=f)
       print(" ",                                      file=f)
       print("Best parameters set found on dev set:",  file=f)
       print(gs.best_params_,                          file=f)


def plot(sc_x, sc_y, x_test, y_test, dataset):
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


def save_model(gs, dataset):
   dump(gs, dataset+'/models/model_MO_'+dataset+'.sav')
