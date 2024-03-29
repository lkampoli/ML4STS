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

n_jobs = -1
trial  = 1

nba = pd.read_csv("solution_XY.dat")
print(type(nba))
print(len(nba))
print(nba.shape)
nba.head()

dataset=np.loadtxt("./solution_XY.dat")
print(dataset.shape)

x = dataset[:,0:50]
y = dataset[:,50:]

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

# DecisionTree
hyper_params = [{'criterion': ('mse',),
                 #'criterion': ('mse', 'friedman_mse', 'mae'),
                 #'splitter': ('best', 'random'),
                 'splitter': ('best',),
                 #'max_features': ('auto', 'sqrt', 'log2'),
                 'max_features': ('auto',),
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
plt.savefig("regression.pdf", dpi=150, crop='false')
plt.show()

# save the model to disk
dump(gs, 'model.sav')
