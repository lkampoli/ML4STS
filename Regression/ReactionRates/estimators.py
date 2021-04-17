#!/usr/bin/env python

from sklearn.tree import DecisionTreeRegressor

from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_hist_gradient_boosting 
from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.multioutput import MultiOutputRegressor, RegressorChain

from sklearn import neighbors
from sklearn.neighbors import KNeighborsRegressor

from sklearn import kernel_ridge
from sklearn.kernel_ridge import KernelRidge

from sklearn.neural_network import MLPRegressor

from sklearn import svm
from sklearn.svm import SVR

from sklearn.utils.fixes import loguniform

#import tensorflow as tf
#from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.layers import Dense, Dropout
#from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
#from tensorflow import keras
#from tensorflow.keras import layers
#from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

#from IPython.display import clear_output
#from livelossplot import PlotLossesKeras
#from keras.callbacks import TensorBoard
#from keras.utils.vis_utils import plot_model
#from keras.models import load_model
#from ann_visualizer.visualize import ann_viz;
#from keras.models import model_from_json
#from keras_sequential_ascii import keras2ascii
#from keras.optimizers import SGD, Adam, RMSprop, Adagrad
#from keras import regularizers


# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor
# criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
# max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, ccp_alpha=0.0
def est_DT():
    hp = [{'criterion': ('mse', 'friedman_mse', 'mae'), 
           'splitter': ('best', 'random'),             
           'max_features': ('auto', 'sqrt', 'log2'),  
    }]
    est = DecisionTreeRegressor()
    return est, hp


# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html?highlight=extra%20tree#sklearn.ensemble.ExtraTreesRegressor
# n_estimators=100, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
# max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=None, random_state=None, 
# verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None
def est_ET():
    hp = [{'n_estimators': (1, 100,),
           'min_weight_fraction_leaf': (0.0, 0.25, 0.5,),
           'max_features': ('sqrt','log2','auto', None,),
           'max_samples': (1,10, 100, 1000,), #loguniform(1, 1000),
           'bootstrap': (True, False,),
           'oob_score': (True, False,),
           'warm_start': (True, False,),
           'criterion': ('mse', 'mae',),
           'max_depth': (1,10,100,None,),
           'max_leaf_nodes': (2, 100,),
           'min_samples_split': (10,),
           'min_samples_leaf': (1,10,100,) #loguniform(1, 100),
    }]
    est = ensemble.ExtraTreesRegressor()
    #regr = MultiOutputRegressor(estimator=est)
    return est, hp


# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html?highlight=gradientboostingregressor#sklearn.ensemble.GradientBoostingRegressor
# loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, 
# min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, 
# max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, 
# tol=0.0001, ccp_alpha=0.0
def est_GB():
    hp = [{'n_estimators': (10, 100, 1000,),
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
    return est, hp


# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html#sklearn.ensemble.HistGradientBoostingRegressor
# loss='least_squares', learning_rate=0.1, max_iter=100, max_leaf_nodes=31, max_depth=None, min_samples_leaf=20, l2_regularization=0.0, 
# max_bins=255, categorical_features=None, monotonic_cst=None, warm_start=False, early_stopping='auto', scoring='loss', 
# validation_fraction=0.1, n_iter_no_change=10, tol=1e-07, verbose=0, random_state=None
def est_HGB(est):
    hp = [{'warm_start': (False, True),
           'max_depth': (1, 10, 100, None,),
           'min_samples_leaf': (2, 5, 10,),
           'loss': ('ls', 'lad', 'huber', 'quantile',),
           'max_leaf_nodes': (2, 10, 20, 30, 40, 50, 100,),
    }]
    est = ensemble.HistGradientBoostingRegressor()
    return est, hp


# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor
# n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None
def est_KN():
    hp = [{'algorithm': ('ball_tree', 'kd_tree', 'brute', 'auto',),
           'n_neighbors': (1, 5, 10, 20),
           'leaf_size': (1, 10, 50, 100,),
           'weights': ('uniform', 'distance',),
#          'metric': ('minkowski', ),  
#          'metric_params': (), 
           'p': (1, 2,),
    }]
    est = neighbors.KNeighborsRegressor()
    return est, hp


# https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html#sklearn.kernel_ridge.KernelRidge
# alpha=1, *, kernel='linear', gamma=None, degree=3, coef0=1, kernel_params=None
def est_KR():
    hp = [{'kernel': ('poly', 'rbf',),
           'alpha': (1e-3, 1e-2, 1e-1, 0.0, 0.5, 1.,),
#          'degree': (),
#          'coef0': (),
           'gamma': (0.1, 1, 2,),
    }]
    est = kernel_ridge.KernelRidge()
    return est, hp


# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html?highlight=mlp#sklearn.neural_network.MLPRegressor
# hidden_layer_sizes=100, activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', 
# learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, 
# warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, 
# beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000
def est_MLP(est):
    hp = [{'hidden_layer_sizes': (10, 50, 100, 150, 200,),
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
    return est, hp


# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
# n_estimators=100, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, 
# min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, 
# min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, 
# verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None
def est_RF():
    hp = [{'n_estimators': (1, 50, 100,),
           'min_weight_fraction_leaf': (0.0, 0.25, 0.5,),
           'max_features': ('sqrt', 'log2', 'auto',),
#          'bootstrap': (True, False,),
#          'oob_score': (True, False,),
#          'warm_start': (True, False,),
           'criterion': ('mse', 'mae',),
           'max_depth': (1, 10, 100,),
           'max_leaf_nodes': (2, 100,),
           'min_samples_split': (2, 5, 10,),
           'min_impurity_decrease': (0.1, 0.2, 0.3,),
           'min_samples_leaf': (1, 10, 100,),
#          'min_impurity_split':= (), 
#          'ccp_alpha': (),
#          'max_samples': (),
    }]
    est = ensemble.RandomForestRegressor(random_state=69)
    return est, hp


# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR
# kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1
def est_SVM():
    """ """
    hp = [{'kernel': ('poly', 'rbf',),
           'gamma': ('scale', 'auto',),
           'C': (1e-1, 1e0, 1e1,),
           'epsilon': (1e-2, 1e-1, 1e0, 1e1,),
           'coef0': (0.0, 0.1, 0.2,),
    }]
    est = svm.SVR()
    return est, hp
