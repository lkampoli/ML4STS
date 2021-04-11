#!/usr/bin/env python
# coding: utf-8

# https://scikit-learn.org/stable/auto_examples/compose/plot_feature_union.html#sphx-glr-auto-examples-compose-plot-feature-union-py
# https://jakevdp.github.io/PythonDataScienceHandbook/06.00-figure-code.html#Principal-Components-Rotation
# https://stackabuse.com/implementing-pca-in-python-with-scikit-learn/

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

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.pipeline import make_pipeline

from yellowbrick.features import PCA

n_jobs = 1
trial  = 1

# Import database
with open('../../../../Data/TCs_air5.txt') as f:
    lines = (line for line in f if not line.startswith('#'))
    dataset = np.loadtxt(lines, skiprows=1)

x = dataset[:,0:7] # T, P, x_N2, x_O2, x_NO, x_N, x_O
y = dataset[:,7:8] # shear

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=69)

sc_x = StandardScaler()
sc_y = StandardScaler()

sc_x.fit(x_train)
x_train = sc_x.transform(x_train)
x_test  = sc_x.transform(x_test)

sc_y.fit(y_train)
y_train = sc_y.transform(y_train)
y_test  = sc_y.transform(y_test)

#dump(sc_x, open('scaler_x_shear.pkl', 'wb'))
#dump(sc_y, open('scaler_y_shear.pkl', 'wb'))
#
#print('Training Features Shape:', x_train.shape)
#print('Training Labels Shape:',   y_train.shape)
#print('Testing Features Shape:',  x_test.shape)
#print('Testing Labels Shape:',    y_test.shape)
#
#hyper_params = [{'criterion': ('mse', 'friedman_mse', 'mae'), # mse
#                 'splitter': ('best', 'random'),              # auto
#                 'max_features': ('auto', 'sqrt', 'log2'),    # best
#}]
#
#est = DecisionTreeRegressor()
#gs  = GridSearchCV(est, cv=10, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2')
#
#t0 = time.time()
#gs.fit(x_train, y_train.ravel())
#runtime = time.time() - t0
#print("Complexity and bandwidth selected and model fitted in %.6f s" % runtime)
#
#train_score_mse  = mean_squared_error(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
#train_score_mae  = mean_absolute_error(sc_y.inverse_transform(y_train),sc_y.inverse_transform(gs.predict(x_train)))
#train_score_evs  = explained_variance_score(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
#train_score_me   = max_error(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
#train_score_msle = mean_squared_log_error(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
#train_score_r2   = r2_score(                sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
#
#test_score_mse  = mean_squared_error(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
#test_score_mae  = mean_absolute_error(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
#test_score_evs  = explained_variance_score(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
#test_score_me   = max_error(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
#test_score_msle = mean_squared_log_error(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
#test_score_r2   = r2_score(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
#
#print()
#print("The model performance for training set")
#print("--------------------------------------")
#print('MAE is      {}'.format(train_score_mae))
#print('MSE is      {}'.format(train_score_mse))
#print('EVS is      {}'.format(train_score_evs))
#print('ME is       {}'.format(train_score_me))
#print('MSLE is     {}'.format(train_score_msle))
#print('R2 score is {}'.format(train_score_r2))
#print()
#print("The model performance for testing set")
#print("--------------------------------------")
#print('MAE is      {}'.format(test_score_mae))
#print('MSE is      {}'.format(test_score_mse))
#print('EVS is      {}'.format(test_score_evs))
#print('ME is       {}'.format(test_score_me))
#print('MSLE is     {}'.format(test_score_msle))
#print('R2 score is {}'.format(test_score_r2))
#print()
#print("Best parameters set found on development set:")
#print(gs.best_params_)
#print()

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

#df = pd.DataFrame({'var':pca.explained_variance_ratio_, 'PC':['PC1','PC2','PC3','PC4']})
#sns.barplot(x='PC',y="var", data=df, color="c");

# Re-train with best parameters
#regr = DecisionTreeRegressor(**gs.best_params_)
regr = DecisionTreeRegressor(criterion='mse', splitter='best', max_features='auto')

#visualizer = PCA(scale=True, proj_features=True)
visualizer = PCA(scale=True, proj_features=True, projection=3)
visualizer.fit_transform(x, y)
visualizer.show()
visualizer.close()

pca = PCA(n_components=1)
x_pca = pca.fit_transform(x)
regr.fit(x_pca,y)

pca_score = pca.explained_variance_ratio_
V = pca.components_

variance = pd.DataFrame(pca.explained_variance_ratio_)
np.cumsum(pca.explained_variance_ratio_)

print(pca.components_)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)

t0 = time.time()
regr.fit(x, y.ravel())
regr_fit = time.time() - t0
print("Complexity and bandwidth selected and model fitted in %.6f s" % regr_fit)

#t0 = time.time()
#y_regr = regr.predict(x_test)
#regr_predict = time.time() - t0
#print("Prediction for %d inputs in %.6f s" % (x_test.shape[0], regr_predict))

t0 = time.time()
regr.fit(x_pca, y.ravel())
regr_fit = time.time() - t0
print("Complexity and bandwidth selected and model fitted in %.6f s" % regr_fit)

#plt.plot(np.cumsum(pca.explained_variance_ratio_))
#plt.xlabel('number of components')
#plt.ylabel('cumulative explained variance');
#plt.show()

# plot data
#plt.scatter(x[:, 0], x[:, 1], alpha=0.2)
#for length, vector in zip(pca.explained_variance_, pca.components_):
#    v = vector * 3 * np.sqrt(length)
#    draw_vector(pca.mean_, pca.mean_ + v)
#plt.axis('equal');

# Fit to data and predict using pipelined GNB and PCA.
unscaled_est = make_pipeline(PCA(n_components=1), regr)
unscaled_est.fit(x_train, y_train)
pred_test = unscaled_est.predict(x_test)

# Fit to data and predict using pipelined scaling, GNB and PCA.
std_est = make_pipeline(StandardScaler(), PCA(n_components=2), regr)
std_est.fit(x_train, y_train)
pred_test_std = std_est.predict(x_test)

# Show prediction accuracies in scaled and unscaled data.
print('\nPrediction accuracy for the normal test dataset with PCA')
test_score_r2 = r2_score(y_test, pred_test)
print('R2 score is {}'.format(test_score_r2))

print('\nPrediction accuracy for the standardized test dataset with PCA')
test_score_r2 = r2_score(y_test, pred_test_std)
print('R2 score is {}'.format(test_score_r2))

# Extract PCA from pipeline
pca = unscaled_est.named_steps['pca']
pca_std = std_est.named_steps['pca']

# Show first principal components
print('\nPC 1 without scaling:\n', pca.components_[0])
print('\nPC 1 with scaling:\n', pca_std.components_[0])

# Use PCA without and with scale on X_train data for visualization.
train_transformed = pca.transform(x_train)
scaler = std_est.named_steps['standardscaler']
x_train_std_transformed = pca_std.transform(scaler.transform(x_train))

#t0 = time.time()
#y_regr = regr.predict(x_test)
#regr_predict = time.time() - t0
#print("Prediction for %d inputs in %.6f s" % (x_test.shape[0], regr_predict))

#importance = regr.feature_importances_
#
## summarize feature importance
#for i,v in enumerate(importance):
#	print('Feature: %0d, Score: %.5f' % (i,v))
#
## plot feature importance
#plt.title("Feature importances")
#features = np.array(['T', 'P', '$X_{N2}$', '$X_{O2}$', '$X_{NO}$', '$X_N$', '$X_O$'])
#plt.bar(features, importance)
##plt.bar([x for x in range(len(importance))], importance)
#plt.savefig("importance_shear.pdf", dpi=150, crop='false')
#plt.show()
#plt.close()
#
#t0 = time.time()
#y_regr = regr.predict(x_test)
#regr_predict = time.time() - t0
#print("Prediction for %d inputs in %.6f s" % (x_test.shape[0], regr_predict))
#
#with open('output.log', 'w') as f:
#    print("Training time: %.6f s" % regr_fit, file=f)
#    print("Prediction time: %.6f s" % regr_predict, file=f)
#    print(" ", file=f)
#    print("The model performance for training set", file=f)
#    print("--------------------------------------", file=f)
#    print('MAE is {}'.format(train_score_mae), file=f)
#    print('MSE is {}'.format(train_score_mse), file=f)
#    print('EVS is {}'.format(train_score_evs), file=f)
#    print('ME is {}'.format(train_score_me), file=f)
#    print('MSLE is {}'.format(train_score_msle), file=f)
#    print('R2 score is {}'.format(train_score_r2), file=f)
#    print(" ", file=f)
#    print("The model performance for testing set", file=f)
#    print("--------------------------------------", file=f)
#    print('MAE is {}'.format(test_score_mae), file=f)
#    print('MSE is {}'.format(test_score_mse), file=f)
#    print('EVS is {}'.format(test_score_evs), file=f)
#    print('ME is {}'.format(test_score_me), file=f)
#    print('MSLE is {}'.format(test_score_msle), file=f)
#    print('R2 score is {}'.format(test_score_r2), file=f)
#    print(" ", file=f)
#    print("Best parameters set found on development set:", file=f)
#    print(gs.best_params_, file=f)
#
#x_test_dim = sc_x.inverse_transform(x_test)
#y_test_dim = sc_y.inverse_transform(y_test)
#y_regr_dim = sc_y.inverse_transform(y_regr)
#
#plt.scatter(x_test_dim[:,0], y_test_dim[:], s=5, c='k', marker='o', label='KAPPA')
#plt.scatter(x_test_dim[:,0], y_regr_dim[:], s=2.5, c='r', marker='o', label='DecisionTree')
#plt.ylabel(r'$\eta$ [Pa·s]')
#plt.xlabel('T [K] ')
#plt.legend()
#plt.tight_layout()
#plt.savefig("shear.pdf", dpi=150, crop='false')
#plt.show()
#plt.close()
#
## save the model to disk
#dump(gs, 'model_shear.sav')
