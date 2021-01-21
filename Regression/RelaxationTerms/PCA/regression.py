#!/usr/bin/env python
# coding: utf-8

# https://scikit-learn.org/stable/auto_examples/compose/plot_feature_union.html#sphx-glr-auto-examples-compose-plot-feature-union-py
# https://jakevdp.github.io/PythonDataScienceHandbook/06.00-figure-code.html#Principal-Components-Rotation
# https://stackabuse.com/implementing-pca-in-python-with-scikit-learn/

import time
import sys
sys.path.insert(0, '../../../Utilities/')

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

from sklearn.decomposition import PCA
#from yellowbrick.features import PCA

from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.pipeline import make_pipeline

n_jobs = 1
trial  = 1

dataset=np.loadtxt("../data/solution_DR.dat")
print(dataset.shape) # 943 x 103

x = dataset[:,0:55]  # x_s[1], time_s[1], Temp[1], rho[1], p[1],
y = dataset[:,55:56]   # RD_mol[47], RD_at[1]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=69)

sc_x = StandardScaler()
sc_y = StandardScaler()

sc_x.fit(x_train)
x_train = sc_x.transform(x_train)
x_test  = sc_x.transform(x_test)

sc_y.fit(y_train)
y_train = sc_y.transform(y_train)
y_test  = sc_y.transform(y_test)

# First verifiy the ideal number of PCA components not to lose much information. 
# Try to retain 90% information, so look where the curve starts to flatten.
# Remember that the n_components must be lower than the number of rows or columns (features)
pca_test = PCA(n_components=55)
pca_test.fit(x_train)
plt.plot(np.cumsum(pca_test.explained_variance_ratio_))
plt.xlabel("Number of components")
plt.ylabel("Cumulative variance")
plt.grid()
plt.savefig("PCA.eps", dpi=150, crop='false')
plt.show()

print("Variance explained by all 55 principal components = ",  sum(pca_test.explained_variance_ratio_ * 100)) 

print(pca_test.explained_variance_ratio_ * 100)
print(np.cumsum(pca_test.explained_variance_ratio_ * 100))

print("Variance explained by the first principal component = ", np.cumsum(pca_test.explained_variance_ratio_ * 100)[0]) 
print("Variance explained by the first 2 principal components = ", np.cumsum(pca_test.explained_variance_ratio_ * 100)[1]) 
print("Variance explained by the first 3 principal component = ", np.cumsum(pca_test.explained_variance_ratio_ * 100)[2]) 

# Pick the optimal number of components. 
# This is how many features we will have for our machine learning
n_PCA_components = 2
pca = PCA(n_components=n_PCA_components)
train_PCA = pca.fit_transform(x_train)
test_PCA = pca.transform(x_test) # Make sure you are just transforming, not fitting 

print(pca.components_)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(np.cumsum(pca.explained_variance_ratio_) )

# If we want 90% information captured we can also try ...
# pca=PCA(0.9)
# principalComponents = pca.fit_transform(X_for_RF)

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

# plot data
#plt.scatter(x_train[:, 0], x_train[:, 1], alpha=0.2)
#for length, vector in zip(pca.explained_variance_, pca.components_):
#    v = vector * 3 * np.sqrt(length)
#    draw_vector(pca.mean_, pca.mean_ + v)
#plt.axis('equal');

#df = pd.DataFrame({'var':pca.explained_variance_ratio_, 'PC':['PC1','PC2','PC3','PC4']})
#sns.barplot(x='PC',y="var", data=df, color="c");

## Re-train with best parameters
##regr = DecisionTreeRegressor(**gs.best_params_)
regr = DecisionTreeRegressor(criterion='mse', splitter='best', max_features='auto')
#
#visualizer = PCA(scale=True, proj_features=True)
##visualizer = PCA(scale=True, proj_features=True, projection=3)
#visualizer.fit_transform(x, y)
#visualizer.show()
##visualizer.close()
#
#pca = PCA(n_components=10)
#x_pca = pca.fit_transform(x)
t0 = time.time()
regr.fit(train_PCA, y_train)
regr_fit = time.time() - t0
print("Total execution time with PCA is %.6f s" % regr_fit)
#
#pca_score = pca.explained_variance_ratio_
#V = pca.components_
#
#variance = pd.DataFrame(pca.explained_variance_ratio_)
#np.cumsum(pca.explained_variance_ratio_)
#
#print(pca.components_)
#print(pca.explained_variance_ratio_)
#print(pca.singular_values_)
#
#t0 = time.time()
#regr.fit(x, y.ravel())
#regr_fit = time.time() - t0
#print("Complexity and bandwidth selected and model fitted in %.6f s" % regr_fit)
#
##t0 = time.time()
##y_regr = regr.predict(x_test)
##regr_predict = time.time() - t0
##print("Prediction for %d inputs in %.6f s" % (x_test.shape[0], regr_predict))
#
t0 = time.time()
pred_test    = regr.predict(test_PCA)

x_test_dim = sc_x.inverse_transform(x_test)
y_test_dim = sc_y.inverse_transform(y_test)
y_regr_dim = sc_y.inverse_transform(pred_test)

#pred_test    = np.argmax(pred_test, axis=1)
#predict_test = le.inverse_transform(pred_test)
pred_time = time.time() - t0
print("Total prediction time with PCA is %.6f s" % pred_time)

#regr.fit(x_pca, y.ravel())
#regr_fit = time.time() - t0
#print("Complexity and bandwidth selected and model fitted in %.6f s" % regr_fit)
#
##plt.plot(np.cumsum(pca.explained_variance_ratio_))
##plt.xlabel('number of components')
##plt.ylabel('cumulative explained variance');
##plt.show()
#
## plot data
##plt.scatter(x[:, 0], x[:, 1], alpha=0.2)
##for length, vector in zip(pca.explained_variance_, pca.components_):
##    v = vector * 3 * np.sqrt(length)
##    draw_vector(pca.mean_, pca.mean_ + v)
##plt.axis('equal');
#
## Fit to data and predict using pipelined GNB and PCA.
#unscaled_est = make_pipeline(PCA(n_components=1), regr)
#unscaled_est.fit(x_train, y_train)
#pred_test = unscaled_est.predict(x_test)
#
## Fit to data and predict using pipelined scaling, GNB and PCA.
#std_est = make_pipeline(StandardScaler(), PCA(n_components=2), regr)
#std_est.fit(x_train, y_train)
#pred_test_std = std_est.predict(x_test)
#
## Show prediction accuracies in scaled and unscaled data.
#print('\nPrediction accuracy for the normal test dataset with PCA')
#test_score_r2 = r2_score(y_test, pred_test)
#print('R2 score is {}'.format(test_score_r2))
#
#print('\nPrediction accuracy for the standardized test dataset with PCA')
#test_score_r2 = r2_score(y_test, pred_test_std)
#print('R2 score is {}'.format(test_score_r2))
#
## Extract PCA from pipeline
#pca = unscaled_est.named_steps['pca']
#pca_std = std_est.named_steps['pca']
#
## Show first principal components
#print('\nPC 1 without scaling:\n', pca.components_[0])
#print('\nPC 1 with scaling:\n', pca_std.components_[0])
#
## Use PCA without and with scale on X_train data for visualization.
#train_transformed = pca.transform(x_train)
#scaler = std_est.named_steps['standardscaler']
#x_train_std_transformed = pca_std.transform(scaler.transform(x_train))
