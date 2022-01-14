#!/usr/bin/env python

# module for time access and conversions
import time

# module for system-specific parameters and functions
import sys
sys.path.insert(0, './')

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
dataset=np.loadtxt("../data/dataset_N2N.dat") #(1936, 289)

print(dataset.shape)

x = dataset[:,0:54]  # x_s, time_s, Temp, ni_n, na_n, rho, v, p
y = dataset[:,54:]   # RDm, RDa, RVTm, RVTa, RVV

print(x.shape)
print(y.shape)

# Scatter plot of the original dataset
plt.scatter(x[:,0], y[:,0], s=2, c='k', marker='o', label='Matlab')
#plt.xscale('log')
#plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.show()

# Initialize layout
#fig, ax = plt.subplots(figsize = (9, 6))
#ax.scatter(x[:,0], y[:,0], s=2, c='k', marker='o', label='Matlab')
#ax.set_xscale("log");
#ax.set_yscale("log");

# Split dataset into train/test
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=69)

### Feature Importance ###
from sklearn.ensemble import RandomForestRegressor

feature_names = [f"feature {i}" for i in range(x.shape[1])]
forest = RandomForestRegressor(max_depth=2, random_state=0)
forest.fit(x_train, y_train)

start_time = time.time()
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

import pandas as pd

forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()

### ###

from sklearn.inspection import permutation_importance

start_time = time.time()
result = permutation_importance(
    forest, x_test, y_test, n_repeats=10, random_state=42, n_jobs=2
)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

forest_importances = pd.Series(result.importances_mean, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()
### End Feature Importance ###


# Standardization
sc_x = StandardScaler()
sc_y = StandardScaler()

# fit scaler
sc_x.fit(x_train)

# transform training dataset
x_train = sc_x.transform(x_train)

# transform test dataset
x_test = sc_x.transform(x_test)

# fit scaler on training dataset
sc_y.fit(y_train)

# transform training dataset
y_train = sc_y.transform(y_train)

# transform test dataset
y_test = sc_y.transform(y_test)

# Save scalers
dump(sc_x, open('scaler_x.pkl', 'wb'))
dump(sc_y, open('scaler_y.pkl', 'wb'))

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:'  , y_train.shape)
print('Testing Features Shape:' , x_test.shape)
print('Testing Labels Shape:'   , y_test.shape)

# Feature selection
# https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/
#from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import f_regression
#from sklearn.feature_selection import SelectFromModel
#from sklearn.feature_selection import mutual_info_regression

# define feature selection
#fs = SelectKBest(score_func=f_regression, k=10)

# apply feature selection
#x_selected = fs.fit_transform(x_train, y_train)
#print(x_selected.shape)

# DecisionTree hyperparameters grid
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor
#                 criterion='mse', 
#                 splitter='best', 
#                 max_depth=None, 
#                 min_samples_split=2, 
#                 min_samples_leaf=1, 
#                 min_weight_fraction_leaf=0.0, 
#                 max_features=None, 
#                 random_state=None, 
#                 max_leaf_nodes=None, 
#                 min_impurity_decrease=0.0, 
#                 min_impurity_split=None, 
#                 ccp_alpha=0.0
hyper_params = [{'criterion': ('mse', 'friedman_mse', 'mae'),
                 'splitter': ('best', 'random'),
                 'max_features': ('auto', 'sqrt', 'log2'),
}]

# Define the estimator
est=DecisionTreeRegressor(random_state=666)

# Exhaustive search over specified parameter values for the estimator
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
gs = GridSearchCV(est, cv=5, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2', 
                  refit=True, pre_dispatch='n_jobs', error_score=np.nan, return_train_score=True)

# Train the model
t0 = time.time()
gs.fit(x_train, y_train)
runtime = time.time() - t0
print("Training time: %.6f s" % runtime)

train_score_mse   = mean_squared_error(      sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_mae   = mean_absolute_error(     sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_evs   = explained_variance_score(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
#train_score_me   = max_error(               sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_r2    = r2_score(                sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
#train_score_msle = mean_squared_log_error(  sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))

test_score_mse   = mean_squared_error(      sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_mae   = mean_absolute_error(     sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_evs   = explained_variance_score(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
#test_score_me   = max_error(               sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
test_score_r2    = r2_score(                sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
#test_score_msle = mean_squared_log_error(  sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))

print()
print("The model performance for training set")
print("--------------------------------------")
print('MAE is      {}'.format(train_score_mae))
print('MSE is      {}'.format(train_score_mse))
print('EVS is      {}'.format(train_score_evs))
#print('ME is      {}'.format(train_score_me))
#print('MSLE is    {}'.format(train_score_msle))
print('R2 score is {}'.format(train_score_r2))
print()
print("The model performance for testing set")
print("--------------------------------------")
print('MAE is      {}'.format(test_score_mae))
print('MSE is      {}'.format(test_score_mse))
print('EVS is      {}'.format(test_score_evs))
#print('ME is      {}'.format(test_score_me))
#print('MSLE is    {}'.format(test_score_msle))
print('R2 score is {}'.format(test_score_r2))
print()
print("Best parameters set found on development set:")
print(gs.best_params_)
print()

# Re-train with best parameters (not needed if Refit=True)
regr = DecisionTreeRegressor(**gs.best_params_)

t0 = time.time()
regr.fit(x_train, y_train)
regr_fit = time.time() - t0
print("Training time: %.6f s" % regr_fit)

importance = regr.feature_importances_

# Summarize feature importance
for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))

# Plot feature importance
plt.title("Feature importance")
features = np.array(["n1","n2","n3","n4","n5","n6","n7","n8","n9","n10","n11","n12","n13","n14","n15","n16","n17","n18","n19","n20","n21","n22","n23","n24","n25","n26","n27","n28","n29","n30","n31","n32","n33","n34","n35","n36","n37","n38","n39","n40","n41","n42","n43","n44","n45","n46","n47","na","V","T"])
plt.bar(features, importance)
plt.xticks(rotation=45)
plt.savefig("importance.pdf", dpi=150, crop='false')
plt.show()
plt.close()

# Prediction
t0 = time.time()
y_regr = regr.predict(x_test)
regr_predict = time.time() - t0
print("Prediction for %d inputs in %.6f s" % (x_test.shape[0], regr_predict))

with open('output.log', 'w') as f:
    print("Training time:   %.6f s" % regr_fit,     file=f)
    print("Prediction time: %.6f s" % regr_predict, file=f)
    print(" ", file=f)
    print("Model performance for training set",     file=f)
    print("--------------------------------------", file=f)
    print('MAE is      {}'.format(train_score_mae), file=f)
    print('MSE is      {}'.format(train_score_mse), file=f)
    print('EVS is      {}'.format(train_score_evs), file=f)
    #print('ME is      {}'.format(train_score_me),  file=f)
    #print('MSLE is    {}'.format(train_score_msle),file=f)
    print('R2 score is {}'.format(train_score_r2),  file=f)
    print(" ",                                      file=f)
    print("Model performance for testing set",      file=f)
    print("--------------------------------------", file=f)
    print('MAE is      {}'.format(test_score_mae),  file=f)
    print('MSE is      {}'.format(test_score_mse),  file=f)
    print('EVS is      {}'.format(test_score_evs),  file=f)
    #print('ME is      {}'.format(test_score_me),   file=f)
    #print('MSLE is    {}'.format(test_score_msle), file=f)
    print('R2 score is {}'.format(test_score_r2),   file=f)
    print(" ",                                      file=f)
    print("Adimensional test metrics",              file=f)
    print("--------------------------------------", file=f)
    print('Mean Absolute Error (MAE):',      mean_absolute_error(y_test, y_regr),          file=f)
    print('Mean Squared Error (MSE):',       mean_squared_error( y_test, y_regr),          file=f)
    print('Root Mean Squared Error (RMSE):', np.sqrt(mean_squared_error( y_test, y_regr)), file=f)
    print(" ",                                      file=f)
    print("Best parameters set found for dev set:", file=f)
    print(gs.best_params_,                          file=f)

print('Mean Absolute Error (MAE):',              mean_absolute_error(y_test, y_regr))
print('Mean Squared Error (MSE):',               mean_squared_error( y_test, y_regr))
print('Root Mean Squared Error (RMSE):', np.sqrt(mean_squared_error( y_test, y_regr)))

# Transform back data
x_test_dim = sc_x.inverse_transform(x_test)
y_test_dim = sc_y.inverse_transform(y_test)
y_regr_dim = sc_y.inverse_transform(y_regr)

# Plot comparison of ground-truth value and prediction
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
plt.legend()
plt.tight_layout()
plt.savefig( "regression.eps", dpi=150, crop='false')
#plt.savefig("regression.pdf", dpi=150, crop='false')
plt.show()

# Parity plot
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_predict.html#
plt.scatter(y_test_dim, y_regr_dim, s=4, c='r', marker='+')
#plt.scatter(y_test_dim[:,0], y_regr_dim[:,0])
#plt.scatter(y_test_dim[:,10],y_regr_dim[:,10], s=2, c='k', marker='o')
#plt.scatter(y_test_dim[:,20],y_regr_dim[:,20], s=2, c='k', marker='o')
#plt.scatter(y_test_dim[:,30],y_regr_dim[:,30], s=2, c='k', marker='o')
#plt.scatter(y_test_dim[:,40],y_regr_dim[:,40], s=2, c='k', marker='o')
#plt.scatter(y_test_dim[:,47],y_regr_dim[:,47], s=2, c='r', marker='+')
plt.plot([y_test_dim.min(), y_test_dim.max()], [y_test_dim.min(), y_test_dim.max()], 'k--', lw=1)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.savefig('parity.eps', dpi=150, crop='true')
plt.show()

# Save the model to disk
dump(gs, 'model.sav')
