#!/usr/bin/env python
# coding: utf-8

# https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/

import time
import sys
sys.path.insert(0, '../../../../../Utilities/')

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

# compare scaling methods for mlp inputs on regression problem
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from matplotlib import pyplot
from numpy import mean
from numpy import std

# prepare dataset with input and output scalers, can be none
def get_dataset(input_scaler, output_scaler):

    # Import database
    with open('../../../../Data/TCs_air5.txt') as f:
        lines = (line for line in f if not line.startswith('#'))
        dataset = np.loadtxt(lines, skiprows=1)

    x = dataset[:,0:1] # T, P, x_N2, x_O2, x_NO, x_N, x_O
    y = dataset[:,7:8] # shear

    # split into train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=69)

    # scale inputs
    if input_scaler is not None:

        # fit scaler
        input_scaler.fit(x_train)

        # transform training dataset
        x_train = input_scaler.transform(x_train)

        # transform test dataset
        x_test = input_scaler.transform(x_test)

    if output_scaler is not None:

        # reshape 1d arrays to 2d arrays
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        # fit scaler on training dataset
        output_scaler.fit(y_train)

        # transform training dataset
        y_train = output_scaler.transform(y_train)

        # transform test dataset
        y_test = output_scaler.transform(y_test)

    return x_train, y_train, x_test, y_test

# fit and evaluate mse of model on test set
def evaluate_model(x_train, y_train, x_test, y_test):

    # define model
    regr = DecisionTreeRegressor(criterion='mse', splitter='best', max_features='auto')

    # fit model
    regr.fit(x_train, y_train.ravel())

    # predict
    test_mse = regr.predict(x_test)
    return test_mse

# evaluate model multiple times with given input and output scalers
def repeated_evaluation(input_scaler, output_scaler, n_repeats=30):

    # get dataset
    x_train, y_train, x_test, y_test = get_dataset(input_scaler, output_scaler)

    # repeated evaluation of model
    results = list()
    for _ in range(n_repeats):
        test_mse = evaluate_model(x_train, y_train, x_test, y_test)
        print(test_mse)
        print(test_mse.shape)
        print(type(test_mse))
        #print('>%.6f' % test_mse)
        results.append(test_mse)
    return results

# unscaled inputs
results_unscaled_inputs = repeated_evaluation(None, StandardScaler())

# normalized inputs
results_normalized_inputs = repeated_evaluation(MinMaxScaler(), StandardScaler())

# standardized inputs
results_standardized_inputs = repeated_evaluation(StandardScaler(), StandardScaler())

# summarize results
print('Unscaled:     %.6f (%.6f)' % (mean(results_unscaled_inputs),     std(results_unscaled_inputs)))
print('Normalized:   %.6f (%.6f)' % (mean(results_normalized_inputs),   std(results_normalized_inputs)))
print('Standardized: %.6f (%.6f)' % (mean(results_standardized_inputs), std(results_standardized_inputs)))

# plot results
results = [results_unscaled_inputs, results_normalized_inputs, results_standardized_inputs]
labels = ['unscaled', 'normalized', 'standardized']
plt.boxplot(results, labels=labels)
plt.show()
