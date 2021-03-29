#!/usr/bin/env python

# https://stackoverflow.com/questions/45074698/how-to-pass-elegantly-sklearns-gridseachcvs-best-parameters-to-another-model
# https://medium.com/@alexstrebeck/training-and-testing-machine-learning-models-e1f27dc9b3cb

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

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score

from sklearn.inspection import permutation_importance

from sklearn import ensemble
from sklearn.ensemble import ExtraTreesRegressor

from joblib import dump, load
import pickle

#from hyperopt import hp, tpe, fmin, rand
#from hyperopt import STATUS_OK
#from hyperopt.pyll.stochastic import sample
#from hyperopt import Trials
#from hyperopt import pyll

#import hpsklearn
#from hpsklearn import HyperoptEstimator
#from hpsklearn import any_preprocessing, pca, standard_scaler, min_max_scaler, normalizer
#from hpsklearn import any_regressor, any_sparse_regressor
#from hpsklearn import any_classifier, any_sparse_classifier
#from hpsklearn import svr, svr_linear, svr_rbf, svr_poly, svr_sigmoid, knn_regression, ada_boost_regression, \
#                      gradient_boosting_regression, random_forest_regression, extra_trees_regression, sgd_regression, \
#                      xgboost_regression

n_jobs = 4
trial  = 1

dataset=np.loadtxt("../data/solution_DR.dat")
print(dataset.shape) # 943 x 103

x = dataset[:,0:55]  # x_s[1], time_s[1], Temp[1], rho[1], p[1],
y = dataset[:,55:]   # RD_mol[47], RD_at[1]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=69)

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

dump(sc_x, open('scaler_x.pkl', 'wb'))
dump(sc_y, open('scaler_y.pkl', 'wb'))

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:',   y_train.shape)
print('Testing Features Shape:',  x_test.shape)
print('Testing Labels Shape:',    y_test.shape)

# Extra Trees
hyper_params = [{'n_estimators': (1, 100,  500, 1000),
                 #'min_weight_fraction_leaf': (0.0, 0.25, 0.5,),
                 'min_weight_fraction_leaf': (0.0,),
                 'max_features': ('sqrt','log2',),
                 #'max_features': ('sqrt','log2','auto',),
                 'max_samples': (1, 10, 100, 200, 300, 500),
                 'bootstrap': (True,),
                 'oob_score': (True,),
                 'warm_start': (True,),
                 'criterion': ('mse',),
                 'max_depth': (None,),
                 #'bootstrap': (True, False,),
                 #'oob_score': (True, False,),
                 #'warm_start': (True, False,),
                 #'criterion': ('mse', 'mae',),
                 #'max_depth': (1, 10, 100, None,),
                 'max_leaf_nodes': (2, 100, 200, 500),
                 #'min_samples_split': (10,),
                 #'min_samples_leaf': (1, 10, 100,),
                 'min_samples_leaf': (1,),
}]

# https://towardsdatascience.com/automated-machine-learning-hyperparameter-tuning-in-python-dfda59b72f8a
# https://towardsdatascience.com/an-introductory-example-of-bayesian-optimization-in-python-with-hyperopt-aae40fff4ff0

#N_FOLDS = 10
#
#def objective(params, n_folds = N_FOLDS):
#    """Objective function for Gradient Boosting Machine Hyperparameter Tuning"""
#
#    # Perform n_fold cross validation with hyperparameters
#    # Use early stopping and evalute based on ROC AUC
#    cv_results = lgb.cv(params, train_set, nfold = n_folds, num_boost_round = 10000,
#                        early_stopping_rounds = 100, metrics = 'auc', seed = 50)
#
#    # Extract the best score
#    best_score = max(cv_results['auc-mean'])
#
#    # Loss must be minimized
#    loss = 1 - best_score
#
#    # Dictionary with information for evaluation
#    return {'loss': loss, 'params': params, 'status': STATUS_OK}
#
## Define the search space
## choice : categorical variables
## quniform : discrete uniform (integers spaced evenly)
## uniform: continuous uniform (floats spaced evenly)
## loguniform: continuous log uniform (floats spaced evenly on a log scale)
#hyper_space = {'class_weight': hp.choice('class_weight', [None, 'balanced']),
#               'boosting_type': hp.choice('boosting_type',
#                                          [{'boosting_type': 'gbdt',
#                                            'subsample': hp.uniform('gdbt_subsample', 0.5, 1)},
#                                           {'boosting_type': 'dart',
#                                            'subsample': hp.uniform('dart_subsample', 0.5, 1)},
#                                           {'boosting_type': 'goss'}]),
#               'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
#               'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
#               'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
#               'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
#               'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
#               'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
#               'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
#}

est=ensemble.ExtraTreesRegressor(random_state=69)
gs = GridSearchCV(est, cv=10, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2')

#hest = hpsklearn.HyperoptEstimator(preprocessing=hpsklearn.components.any_preprocessing('pp'), #[ pca('my_pca') ],
#                                   regressor=hpsklearn.components.extra_trees_regression('ET'),
#                                   algo=tpe.suggest,
#                                   trial_timeout=15.0, # seconds
#                                   max_evals=30,
#                                   seed=69,)
#
#hest.fit( x_train, y_train )
#print( hest.score( x_test, y_test ) )
#print( hest.best_model() )
#
#hest.retrain_best_model_on_full_data(x_train, y_train)
#print('Test R2:', hest.score(x_test, y_test))

t0 = time.time()
gs.fit(x_train, y_train)
runtime = time.time() - t0
print("Complexity and bandwidth selected and model fitted in %.6f s" % runtime)



train_score_mse = mean_squared_error(      sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_mae = mean_absolute_error(     sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_evs = explained_variance_score(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
#train_score_me  = max_error(               sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
train_score_r2  = r2_score(                sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))

test_score_mse  = mean_squared_error(      sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))
test_score_mae  = mean_absolute_error(     sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))
test_score_evs  = explained_variance_score(sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))
#test_score_me   = max_error(               sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))
test_score_r2   = r2_score(                sc_y.inverse_transform(y_test),  sc_y.inverse_transform(gs.predict(x_test)))

print()
print("The model performance for training set")
print("--------------------------------------")
print('MAE is {}'.format(train_score_mae))
print('MSE is {}'.format(train_score_mse))
print('EVS is {}'.format(train_score_evs))
#print('ME is {}'.format(train_score_me))
print('R2 score is {}'.format(train_score_r2))
print()
print("The model performance for testing set")
print("--------------------------------------")
print('MAE is {}'.format(test_score_mae))
print('MSE is {}'.format(test_score_mse))
print('EVS is {}'.format(test_score_evs))
#print('ME is {}'.format(test_score_me))
print('R2 score is {}'.format(test_score_r2))
print()
print("Best parameters set found on development set:")
print(gs.best_params_)
print()

# Re-train with best parameters
regr = ExtraTreesRegressor(**gs.best_params_, random_state=69)

t0 = time.time()
regr.fit(x_train, y_train)
regr_fit = time.time() - t0
print("Complexity and bandwidth selected and model fitted in %.6f s" % regr_fit)

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
   # print('ME is {}'.format(train_score_me), file=f)
    print('R2 score is {}'.format(train_score_r2), file=f)
    print(" ", file=f)
    print("The model performance for testing set", file=f)
    print("--------------------------------------", file=f)
    print('MAE is {}'.format(test_score_mae), file=f)
    print('MSE is {}'.format(test_score_mse), file=f)
    print('EVS is {}'.format(test_score_evs), file=f)
    #print('ME is {}'.format(test_score_me), file=f)
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

plt.scatter(y_test_dim, y_regr_dim, s=4, c='r', marker='+')
plt.plot([y_test_dim.min(), y_test_dim.max()], [y_test_dim.min(), y_test_dim.max()], 'k--', lw=1)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.savefig('parity.eps', dpi=150, crop='true')
plt.show()

plt.scatter(x_test_dim[:,0], y_test_dim[:,0], s=2, c='k', marker='o', label='Matlab')
plt.scatter(x_test_dim[:,0], y_regr_dim[:,0], s=2, c='r', marker='+', label='ExtraTrees')
#plt.ylabel(r'$\eta$ [Pa·s]')
plt.xlabel('T [K] ')
plt.legend()
plt.tight_layout()
plt.savefig("regression.pdf", dpi=150, crop='false')
plt.show()
plt.close()

# save the model to disk
dump(gs, 'model.sav')