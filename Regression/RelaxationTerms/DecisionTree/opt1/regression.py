#!/usr/bin/env python

import time
import sys
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

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, cross_val_score

from sklearn.tree import DecisionTreeRegressor

import pickle
from joblib import dump, load

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

n_jobs = -1
trial  = 1

dataset=np.loadtxt("./solution_ODE_XY.dat")

print(dataset.shape)

x = dataset[:,0:1]
y = dataset[:,1:]

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

dump(sc_x, open('scaler_x.pkl', 'wb'))
dump(sc_y, open('scaler_y.pkl', 'wb'))

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:'  , y_train.shape)
print('Testing Features Shape:' , x_test.shape)
print('Testing Labels Shape:'   , y_test.shape)

# https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import mutual_info_regression

# define feature selection
#fs = SelectKBest(score_func=f_regression, k=10)

# apply feature selection
#x_selected = fs.fit_transform(x_train, y_train)
#print(x_selected.shape)

# DecisionTree
hyper_params = [{#'criterion': ('mse',),
                 'criterion': ('mse', 'friedman_mse', 'mae'),
                 'splitter': ('best', 'random'),
                 #'splitter': ('best',),
                 'max_features': ('auto', 'sqrt', 'log2'),
                 #'max_features': ('auto',),
}]

est=DecisionTreeRegressor(random_state=69)
gs = GridSearchCV(est, cv=10, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2')

t0 = time.time()
gs.fit(x_train, y_train)
runtime = time.time() - t0
print("Complexity and bandwidth selected and model fitted in %.6f s" % runtime)

train_score_mse  = mean_squared_error(      sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_mae  = mean_absolute_error(     sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_evs  = explained_variance_score(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
#train_score_me  = max_error(               sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_r2   = r2_score(                sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
#train_score_msle = mean_squared_log_error(  sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))

test_score_mse  = mean_squared_error(      sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))
test_score_mae  = mean_absolute_error(     sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))
test_score_evs  = explained_variance_score(sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))
#test_score_me  = max_error(               sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))
test_score_r2   = r2_score(                sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))
#test_score_msle = mean_squared_log_error(  sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))

print()
print("The model performance for training set")
print("--------------------------------------")
print('MAE is      {}'.format(train_score_mae))
print('MSE is      {}'.format(train_score_mse))
print('EVS is      {}'.format(train_score_evs))
#print('ME is       {}'.format(train_score_me))
#print('MSLE is     {}'.format(train_score_msle))
print('R2 score is {}'.format(train_score_r2))
print()
print("The model performance for testing set")
print("--------------------------------------")
print('MAE is      {}'.format(test_score_mae))
print('MSE is      {}'.format(test_score_mse))
print('EVS is      {}'.format(test_score_evs))
#print('ME is       {}'.format(test_score_me))
#print('MSLE is     {}'.format(test_score_msle))
print('R2 score is {}'.format(test_score_r2))
print()
print("Best parameters set found on development set:")
print(gs.best_params_)
print()

# Re-train with best parameters
regr = DecisionTreeRegressor(**gs.best_params_, random_state=69)

t0 = time.time()
regr.fit(x_train, y_train)
regr_fit = time.time() - t0
print("Complexity and bandwidth selected and model fitted in %.6f s" % regr_fit)

importance = regr.feature_importances_

# summarize feature importance
for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))

# plot feature importance
plt.title("Feature importances")
#features = np.array(['T', 'P', '$X_{N2}$', '$X_{O2}$', '$X_{NO}$', '$X_N$', '$X_O$'])
#plt.bar(features, importance)
plt.bar([x for x in range(len(importance))], importance)
plt.savefig("importance.pdf", dpi=150, crop='false')
plt.show()
plt.close()

# https://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection
#model = SelectFromModel(regr, prefit=True)
#x_new = model.transform(x_train)
#print(x_new.shape)

#from sklearn.metrics import r2_score
#from rfpimp import permutation_importances
#
#def r2(regr, x_train, y_train):
#    return r2_score(y_train, regr.predict(x_train))
#
#perm_imp_rfpimp = permutation_importances(regr, x_train, y_train, r2)
#print(perm_imp_rfpimp)

# https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
#importances = regr.feature_importances_
#std = np.std([regr.feature_importances_ for tree in regr.estimators_], axis=0)
#indices = np.argsort(importances)[::-1]
#
## Print the feature ranking
#print("Feature ranking:")
#
#for f in range(x.shape[1]):
#    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
#
## Plot the impurity-based feature importances of the forest
#plt.figure()
#plt.title("Feature importances")
#plt.bar(range(x.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
#plt.xticks(range(x.shape[1]), indices)
#plt.xlim([-1, x.shape[1]])
#plt.show()

t0 = time.time()
y_regr = regr.predict(x_test)
regr_predict = time.time() - t0
print("Prediction for %d inputs in %.6f s" % (x_test.shape[0], regr_predict))

with open('output.log', 'w') as f:
    print("Training time: %.6f s" % regr_fit, file=f)
    print("Prediction time: %.6f s" % regr_predict, file=f)
    print(" ", file=f)
    print("The model performance for training set", file=f)
    print("--------------------------------------", file=f)
    print('MAE is {}'.format(train_score_mae), file=f)
    print('MSE is {}'.format(train_score_mse), file=f)
    print('EVS is {}'.format(train_score_evs), file=f)
    #print('ME is {}'.format(train_score_me), file=f)
    #print('MSLE is {}'.format(train_score_msle), file=f)
    print('R2 score is {}'.format(train_score_r2), file=f)
    print(" ", file=f)
    print("The model performance for testing set", file=f)
    print("--------------------------------------", file=f)
    print('MAE is {}'.format(test_score_mae), file=f)
    print('MSE is {}'.format(test_score_mse), file=f)
    print('EVS is {}'.format(test_score_evs), file=f)
    #print('ME is {}'.format(test_score_me), file=f)
    #print('MSLE is {}'.format(test_score_msle), file=f)
    print('R2 score is {}'.format(test_score_r2), file=f)
    print(" ", file=f)
    print("Adimensional test metrics", file=f)
    print("--------------------------------------", file=f)
    print('Mean Absolute Error (MAE):',              mean_absolute_error(y_test, y_regr), file=f)
    print('Mean Squared Error (MSE):',               mean_squared_error(y_test, y_regr), file=f)
    print('Root Mean Squared Error (RMSE):', np.sqrt(mean_squared_error(y_test, y_regr)), file=f)
    print(" ", file=f)
    print("Best parameters set found on development set:", file=f)
    print(gs.best_params_, file=f)

print('Mean Absolute Error (MAE):',              mean_absolute_error(y_test, y_regr))
print('Mean Squared Error (MSE):',               mean_squared_error(y_test, y_regr))
print('Root Mean Squared Error (RMSE):', np.sqrt(mean_squared_error(y_test, y_regr)))

x_test_dim = sc_x.inverse_transform(x_test)
y_test_dim = sc_y.inverse_transform(y_test)
y_regr_dim = sc_y.inverse_transform(y_regr)

plt.scatter(x_test_dim[:,0], y_test_dim[:,0], s=2, c='k', marker='o', label='Matlab')
plt.scatter(x_test_dim[:,0], y_regr_dim[:,0], s=2, c='r', marker='+', label='DecisionTree')
#plt.scatter(x_test_dim[:,10], y_test_dim[:,10], s=2, c='k', marker='o')
#plt.scatter(x_test_dim[:,10], y_regr_dim[:,10], s=2, c='r', marker='+')
#plt.scatter(x_test_dim[:,20], y_test_dim[:,20], s=2, c='k', marker='o')
#plt.scatter(x_test_dim[:,20], y_regr_dim[:,20], s=2, c='r', marker='+')
#plt.scatter(x_test_dim[:,30], y_test_dim[:,30], s=2, c='k', marker='o')
#plt.scatter(x_test_dim[:,30], y_regr_dim[:,30], s=2, c='r', marker='+')
#plt.scatter(x_test_dim[:,40], y_test_dim[:,40], s=2, c='k', marker='o')
#plt.scatter(x_test_dim[:,40], y_regr_dim[:,40], s=2, c='r', marker='+')
#plt.scatter(x_test_dim[:,47], y_test_dim[:,47], s=2, c='k', marker='o')
#plt.scatter(x_test_dim[:,47], y_regr_dim[:,47], s=2, c='r', marker='+')
#plt.title('Relaxation term $R_{ci}$ regression')
plt.ylabel('$R_{ci}$ $[J/m^3/s]$')
plt.xlabel('T [K] ')
#set_xticks(np.arange(0, 55, 1))
#plt.set_xticklabels(np.arange(0, 10, 0.5))
plt.legend()
plt.tight_layout()
#plt.savefig("regression_DT.eps", dpi=150, crop='false')
#plt.savefig("regression_DT.pdf", dpi=150, crop='false')
plt.show()

# save the model to disk
dump(gs, 'model.sav')

## https://gdcoder.com/decision-tree-regressor-explained-in-depth/
#from sklearn.tree import export_graphviz
#import IPython, graphviz, re, math
#
#def draw_tree(t, col_names, size=9, ratio=0.5, precision=3):
#    """ Draws a representation of a random forest in IPython.
#    Parameters:
#    -----------
#    t: The tree you wish to draw
#    df: The data used to train the tree. This is used to get the names of the features.
#    """
#    s=export_graphviz(t, out_file=None, feature_names=col_names, filled=True,
#                      special_characters=True, rotate=True, precision=precision)
#    IPython.display.display(graphviz.Source(re.sub('Tree {',
#       f'Tree {{ size={size}; ratio={ratio}',s)))
#col_names =['X']
##draw_tree(dt, col_names, precision=3)
#
#from yellowbrick.features import PCA

#visualizer = PCA(scale=True, proj_features=True)
#visualizer = PCA(scale=True, proj_features=True, projection=3)
#visualizer.fit_transform(x, y)
#visualizer.show()
#visualizer.close()

# https://www.codementor.io/@divyeshaegis/when-to-use-pca-before-or-after-a-train-test-split-vxdrlu6ci

#pca = PCA(n_components = 10)
#x_train_pca = pca.fit_transform(x_train)
#print("original shape:   ", x_train.shape)
#print("transformed shape:", x_train_pca.shape)
#
#print(pca.explained_variance_)
#print(pca.components_)
#print(pca.explained_variance_ratio_)
#print(pca.singular_values_)
#
#pca.fit(x_test)
#x_test_pca = pca.transform(x_test)
#print("original shape:   ", x_test.shape)
#print("transformed shape:", x_test_pca.shape)
#
#print(pca.explained_variance_)
#print(pca.components_)
#print(pca.explained_variance_ratio_)
#print(pca.singular_values_)
#
#plt.figure()
#plt.plot(np.cumsum(pca.explained_variance_ratio_))
#plt.xlabel('Number of Components')
#plt.ylabel('Variance (%)')
#plt.title('Explained Variance')
#plt.show()

#pca = PCA(n_components=10)
#x_train_pca = pca.fit_transform(x_train)
#regr.fit(x_train_pca,y_train)
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

# https://machinelearningmastery.com/dimensionality-reduction-algorithms-with-python/
# define the pipeline
#steps = [('pca', PCA(n_components=10)), ('m', regr)]
#regr  = Pipeline(steps=steps)

#pca = PCA(n_components=2)
#pca.fit(x_train)
#
#print(pca.components_)
#print(pca.explained_variance_)

#def plot_components(c, v0, v1, ax=None):
#    ax = ax or plt.gca()
#    ax.annotate(str(c)+'component',v0,v1, arrowprops=dict(arrowstyle='->',facecolor='black', shrinkA=0, shrinkB=0),)

# plot data
#plt.scatter(x_train[:,], X_train.loc[:,'BILL_AMT6'], alpha=0.2)
#i = 0
#for length, vector in zip(pca.explained_variance_, pca.components_):
#   i = i+1
#   v = vector * 3 * np.sqrt(length)
#   plot_components(i,pca.mean_, pca.mean_ + v)
#plt.axis('equal');
