#!/usr/bin/env python

# module for time access and conversions
import time

# module for system-specific parameters and functions
import sys
sys.path.insert(0, './')

#from plotting import newfig, savefig
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import pandas as pd
import seaborn as sns

import operator
import itertools

### scikit-learn modules ###
from sklearn import metrics
from sklearn.metrics import *

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, cross_val_score

from sklearn.tree import DecisionTreeRegressor

#from sklearn.pipeline import Pipeline
#from sklearn.decomposition import PCA

# module for python object serialization
import pickle

# module for lightweight pipelining
from joblib import dump, load

# define the number of jobs = number of cores
n_jobs = 2

# Load dataset
#dataset = [x_s, time_s, Temp, ni_n, na_n, rho, v, p, E, H, rhs];
dataset=np.loadtxt("./data/dataset_N2N_rhs.dat.OK") #(1936, ?289)

x = dataset[:,0:56]  # x_s, time_s, Temp, ni_n, na_n, rho, v, p, E, H
y = dataset[:,56:]   # rhs

print(dataset.shape)
print(x.shape)
print(y.shape)

# Scatter plot of the original dataset
plt.scatter(x[:,2], y[:,3], s=2, c='k', marker='+', label='Matlab')
plt.scatter(x[:,2], y[:,4], s=2, c='b', marker='+', label='Matlab')
plt.scatter(x[:,2], y[:,5], s=2, c='r', marker='+', label='Matlab')
plt.scatter(x[:,2], y[:,6], s=2, c='g', marker='+', label='Matlab')
plt.scatter(x[:,2], y[:,9], s=2, c='y', marker='+', label='Matlab')
#plt.xscale('log')
#plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.show()
