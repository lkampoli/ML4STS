import numpy as np
import pandas as pd

import time

from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.utils.fixes import loguniform


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


def getDefaultAccuracy(model, params, X, y):
    from sklearn.model_selection import cross_val_score, KFold
    grid = cross_val_score(model,
                           X, 
                           y,
#                          cv=KFold(n_splits=1, shuffle=True, random_state=666,
                           scoring='r2')
#   rmse_scores = np.sqrt(-grid)
    return display_scores(grid)


def get_GridSearchCV(model, params, X, y):
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    from sklearn.model_selection import GridSearchCV
    print("performing grid search CV ...")
    grid = GridSearchCV(estimator=model,        # the model to grid search
                        param_grid=params,      # the parameter set to try
                        error_score=0.,         # if a parameter set raises an error, 
#                       cv=1,                   # continue and set the performance to 0
#                       refit=True, 
#                       pre_dispatch='n_jobs',
#                       return_train_score=True,
                        verbose=2, 
                        n_jobs=1,
                        scoring='r2'
                        )

    grid.fit(X, y.ravel()) # fit the model and parameters

    # our classical metric for performance
    print("Best Accuracy: {}".format(grid.best_score_))

    # the best parameters that caused the best accuracy
    print("Best Parameters: {}".format(grid.best_params_))
    
    # the average time it took a model to fit to the data (in seconds)
    print("Average Time to Fit (s): {}".format(round(grid.cv_results_['mean_fit_time'].mean(), 3)))
    
    # the average time it took a model to predict out of sample data (in seconds)
    # this metric gives us insight into how this model will perform in real-time analysis
    print("Average Time to Score (s): {}".format(round(grid.cv_results_['mean_score_time'].mean(), 3)))

    print(pd.DataFrame(grid.cv_results_).sort_values("mean_test_score", ascending=False).head())


def get_RandomizedGridSearchCV(model, params, X, y):
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
    print("performing randomized grid search CV ...")
    grid = RandomizedSearchCV(estimator=model,
                              param_distributions=params,
                              n_iter=250,
                              scoring="r2", # 'neg_mean_squared_error'
#                             cv=StratifiedKFold(n_splits=2),
                              verbose=2,
                              random_state=666,
                             )

    grid.fit(X, y.ravel()) # fit the model and parameters

    # our classical metric for performance
    print("Best Accuracy: {}".format(grid.best_score_))

    # the best parameters that caused the best accuracy
    print("Best Parameters: {}".format(grid.best_params_))
    
    # the average time it took a model to fit to the data (in seconds)
    #print("Average Time to Fit (s): {}".format(round(grid.cv_results_['mean_fit_time'].mean(), 3)))
    
    # the average time it took a model to predict out of sample data (in seconds)
    # this metric gives us insight into how this model will perform in real-time analysis
    #print("Average Time to Score (s): {}".format(round(grid.cv_results_['mean_score_time'].mean(), 3)))

    print(pd.DataFrame(grid.cv_results_).sort_values("mean_test_score", ascending=False).head())


def get_GeneticGridSearchCV(model, params, X, y):
    from evolutionary_search import EvolutionaryAlgorithmSearchCV
    print("performing genetic grid search ...")
    grid = EvolutionaryAlgorithmSearchCV(estimator=model,
                                         params=params,
                                         scoring="r2",
#                                        cv=StratifiedKFold(n_splits=2),
                                         verbose=True,
                                         population_size=50,
                                         gene_mutation_prob=0.10,
                                         tournament_size=3,
                                         generations_number=10,
#                                        pmap = pool.map,
                                        )

    grid.fit(X, y.ravel()) # fit the model and parameters

    # our classical metric for performance
    print("Best Accuracy: {}".format(grid.best_score_))

    # the best parameters that caused the best accuracy
    print("Best Parameters: {}".format(grid.best_params_))
    
    # the average time it took a model to fit to the data (in seconds)
    print("Average Time to Fit (s): {}".format(round(grid.cv_results_['mean_fit_time'].mean(), 3)))
    
    # the average time it took a model to predict out of sample data (in seconds)
    # this metric gives us insight into how this model will perform in real-time analysis
    print("Average Time to Score (s): {}".format(round(grid.cv_results_['mean_score_time'].mean(), 3)))

    print(pd.DataFrame(grid.cv_results_).sort_values("mean_test_score", ascending=False).head())


# Set up some parameters for our grid search machine learning models


# Decision Tree
from sklearn.tree import DecisionTreeRegressor
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor
# criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
# max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, ccp_alpha=0.0
DecisionTreeParameters = { 'criterion': ('mse', 'friedman_mse', 'mae'),
                           'splitter': ('best', 'random'),
                           'max_features': ('auto', 'sqrt', 'log2'),
                         }

DecisionTreeParameters_pipe = { 'regressor__criterion': ('mse', 'friedman_mse', 'mae'),
                                'regressor__splitter': ('best', 'random'),
                                'regressor__max_features': ('auto', 'sqrt', 'log2'),
                              }


# Extra Trees
from sklearn.ensemble import ExtraTreesRegressor
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html?highlight=extra%20tree#sklearn.ensemble.ExtraTreesRegressor
# n_estimators=100, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
# max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=None, random_state=None,
# verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None
ExtraTreeParameters = { 'n_estimators': (1, 100,),
                        'min_weight_fraction_leaf': (0.0, 0.25, 0.5,),
                        'max_features': ('sqrt','log2','auto', None,),
                        'max_samples': loguniform(1, 1000),
                        'bootstrap': (True, False,),
                        'oob_score': (True, False,),
                        'warm_start': (True, False,),
                        'criterion': ('mse', 'mae',),
                        'max_depth': (1,10,100,None,),
                        'max_leaf_nodes': (2, 100,),
                        'min_samples_split': (10,),
                        'min_samples_leaf': loguniform(1, 100),
                      }

ExtraTreeParameters_pipe = { 'regressor__n_estimators': (1, 100,),
                             'regressor__min_weight_fraction_leaf': (0.0, 0.25, 0.5,),
                             'regressor__max_features': ('sqrt','log2','auto', None,),
                             'regressor__max_samples': loguniform(1, 1000),
                             'regressor__bootstrap': (True, False,),
                             'regressor__oob_score': (True, False,),
                             'regressor__warm_start': (True, False,),
                             'regressor__criterion': ('mse', 'mae',),
                             'regressor__max_depth': (1,10,100,None,),
                             'regressor__max_leaf_nodes': (2, 100,),
                             'regressor__min_samples_split': (10,),
                             'regressor__min_samples_leaf': loguniform(1, 100),
                           }


# Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html?highlight=gradientboostingregressor#sklearn.ensemble.GradientBoostingRegressor
# loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1,
# min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None,
# max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None,
# tol=0.0001, ccp_alpha=0.0
GradientBoostingParameters = { 'n_estimators': (10, 100, 1000,),
                               'min_weight_fraction_leaf': (0.0, 0.1, 0.2, 0.3,),
                               'max_features': ('sqrt', 'log2', 'auto',),
                               'warm_start': (False, True),
                               'criterion': ('friedman_mse', 'mse', 'mae',),
                               'max_depth': (1, 10, 100, None,),
                               'min_samples_split': (2, 5, 10,),
                               'min_samples_leaf': (2, 5, 10,),
                               'loss': ('ls', 'lad', 'huber', 'quantile',),
                             }

GradientBoostingParameters_pipe = { 'regressor__n_estimators': (10, 100, 1000,),
                                    'regressor__min_weight_fraction_leaf': (0.0, 0.1, 0.2, 0.3,),
                                    'regressor__max_features': ('sqrt', 'log2', 'auto',),
                                    'regressor__warm_start': (False, True),
                                    'regressor__criterion': ('friedman_mse', 'mse', 'mae',),
                                    'regressor__max_depth': (1, 10, 100, None,),
                                    'regressor__min_samples_split': (2, 5, 10,),
                                    'regressor__min_samples_leaf': (2, 5, 10,),
                                    'regressor__loss': ('ls', 'lad', 'huber', 'quantile',),
                                  }


# Hist Gradient Boosting
from sklearn.ensemble import HistGradientBoostingRegressor
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html#sklearn.ensemble.HistGradientBoostingRegressor
# loss='least_squares', learning_rate=0.1, max_iter=100, max_leaf_nodes=31, max_depth=None, min_samples_leaf=20, l2_regularization=0.0,
# max_bins=255, categorical_features=None, monotonic_cst=None, warm_start=False, early_stopping='auto', scoring='loss',
# validation_fraction=0.1, n_iter_no_change=10, tol=1e-07, verbose=0, random_state=None
HistGradientBoostingParameters = { 'warm_start': (False, True),
                                   'max_depth': (1, 10, 100, None,),
                                   'min_samples_leaf': (2, 5, 10,),
                                   'loss': ('ls', 'lad', 'huber', 'quantile',),
                                   'max_leaf_nodes': (2, 10, 20, 30, 40, 50, 100,),
                                 }

HistGradientBoostingParameters_pipe = { 'regressor__warm_start': (False, True),
                                        'regressor__max_depth': (1, 10, 100, None,),
                                        'regressor__min_samples_leaf': (2, 5, 10,),
                                        'regressor__loss': ('ls', 'lad', 'huber', 'quantile',),
                                        'regressor__max_leaf_nodes': (2, 10, 20, 30, 40, 50, 100,),
                                      }


# k-Nearest Neighbors
from sklearn.neighbors import KNeighborsRegressor
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor
# n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None
KNeighborsParameters = { 'algorithm': ('ball_tree', 'kd_tree', 'brute', 'auto',),
                         'n_neighbors': (1, 5, 10, 20),
                         'leaf_size': (1, 10, 50, 100,),
                         'weights': ('uniform', 'distance',),
                         'metric': ('minkowski', ),
                         'metric_params': (),
                         'p': (1, 2,),
                       }

KNeighborsParameters_pipe = { 'regressor__algorithm': ('ball_tree', 'kd_tree', 'brute', 'auto',),
                              'regressor__n_neighbors': (1, 5, 10, 20),
                              'regressor__leaf_size': (1, 10, 50, 100,),
                              'regressor__weights': ('uniform', 'distance',),
                              'regressor__metric': ('minkowski', ),
                              'regressor__metric_params': (),
                              'regressor__p': (1, 2,),
                            }


# Kernel Ridge
from sklearn.kernel_ridge import KernelRidge
# https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html#sklearn.kernel_ridge.KernelRidge
# alpha=1, *, kernel='linear', gamma=None, degree=3, coef0=1, kernel_params=None
KernelRidgeParameters = { 'kernel': ('poly', 'rbf',),
                          'alpha': (1e-3, 1e-2, 1e-1, 0.0, 0.5, 1.,),
#                         'degree': (),
#                         'coef0': (),
                          'gamma': (0.1, 1, 2,),
                        }

KernelRidgeParameters_pipe = { 'regressor__kernel': ('poly', 'rbf',),
                               'regressor__alpha': (1e-3, 1e-2, 1e-1, 0.0, 0.5, 1.,),
#                              'regressor__degree': (),
#                              'regressor__coef0': (),
                               'regressor__gamma': (0.1, 1, 2,),
                             }


# Multi-Layer Perceptron
from sklearn.neural_network import MLPRegressor
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html?highlight=mlp#sklearn.neural_network.MLPRegressor
# hidden_layer_sizes=100, activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant',
# learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False,
# warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1,
# beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000
MultiLayerPerceptronarameters = { 'hidden_layer_sizes': (10, 50, 100, 150, 200,),
                                  'activation' : ('tanh', 'relu',),
                                  'solver' : ('lbfgs','adam','sgd',),
                                  'learning_rate' : ('constant', 'invscaling', 'adaptive',),
                                  'nesterovs_momentum': (True, False,),
                                  'alpha': (0.00001, 0.0001, 0.001, 0.01, 0.1, 0.0,),
                                  'warm_start': (True, False,),
                                  'early_stopping': (True, False,),
                                  'max_iter': (1000,)
                                }

MultiLayerPerceptronarameters_pipe = { 'regressor__hidden_layer_sizes': (10, 50, 100, 150, 200,),
                                       'regressor__activation' : ('tanh', 'relu',),
                                       'regressor__solver' : ('lbfgs','adam','sgd',),
                                       'regressor__learning_rate' : ('constant', 'invscaling', 'adaptive',),
                                       'regressor__nesterovs_momentum': (True, False,),
                                       'regressor__alpha': (0.00001, 0.0001, 0.001, 0.01, 0.1, 0.0,),
                                       'regressor__warm_start': (True, False,),
                                       'regressor__early_stopping': (True, False,),
                                       'regressor__max_iter': (1000,)
                                     }


# Random Forest
from sklearn.ensemble import RandomForestRegressor
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
# n_estimators=100, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1,
# min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
# min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None,
# verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None
RandomForestParameters = { 'n_estimators': (1, 50, 100,),
                           'min_weight_fraction_leaf': (0.0, 0.25, 0.5,),
                           'max_features': ('sqrt', 'log2', 'auto',),
#                          'bootstrap': (True, False,),
#                          'oob_score': (True, False,),
#                          'warm_start': (True, False,),
                           'criterion': ('mse', 'mae',),
                           'max_depth': (1, 10, 100,),
                           'max_leaf_nodes': (2, 100,),
                           'min_samples_split': (2, 5, 10,),
                           'min_impurity_decrease': (0.1, 0.2, 0.3,),
                           'min_samples_leaf': (1, 10, 100,),
#                          'min_impurity_split':= (),
#                          'ccp_alpha': (),
#                          'max_samples': (),
                         }

RandomForestParameters_pipe = { 'regressor__n_estimators': (1, 50, 100,),
                                'regressor__min_weight_fraction_leaf': (0.0, 0.25, 0.5,),
                                'regressor__max_features': ('sqrt', 'log2', 'auto',),
#                               'regressor__bootstrap': (True, False,),
#                               'regressor__oob_score': (True, False,),
#                               'regressor__warm_start': (True, False,),
                                'regressor__criterion': ('mse', 'mae',),
                                'regressor__max_depth': (1, 10, 100,),
                                'regressor__max_leaf_nodes': (2, 100,),
                                'regressor__min_samples_split': (2, 5, 10,),
                                'regressor__min_impurity_decrease': (0.1, 0.2, 0.3,),
                                'regressor__min_samples_leaf': (1, 10, 100,),
#                               'regressor__min_impurity_split':= (),
#                               'regressor__ccp_alpha': (),
#                               'regressor__max_samples': (),
                              }


# Support Vector Machine
from sklearn.svm import SVR
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR
# kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1
SupportVectorMachineParameters = { 'kernel' : ('poly', 'rbf',),
                                   'gamma'  : ('scale', 'auto',),
                                   'C'      : np.logspace(-9, 9, num=25, base=10),
                                   'epsilon': (1e-2, 1e-1, 1e0, 1e1,),
                                   'coef0'  : (0.0, 0.1, 0.2,),
                                 }

SupportVectorMachineParameters_pipe = { 'regressor__kernel' : ('poly', 'rbf',),
                                        'regressor__gamma'  : ('scale', 'auto',),
                                        'regressor__C'      : (1e-1, 1e0, 1e1,),
                                        'regressor__epsilon': (1e-2, 1e-1, 1e0, 1e1,),
                                        'regressor__coef0'  : (0.0, 0.1, 0.2,),
                                      }


# Import dataset
dataset=np.loadtxt("./transposed_reshaped_data.txt.old")
X = dataset[:,0:50]  # ni_n[47], na_n[1], V, T
y = dataset[:,50:51] # RD_mol[47], RD_at[1]
print(dataset.shape)
print(X.shape)
print(y.shape)


# Instantiate the machine learning models
SupportVectorMachine  = SVR()
KernelRidge           = KernelRidge()
MultiLayerPerceptron  = MLPRegressor()
KNeighbors            = KNeighborsRegressor()
ExtraTree             = ExtraTreesRegressor()
DecisionTree          = DecisionTreeRegressor()
RandomForest          = RandomForestRegressor()
GradientBoosting      = GradientBoostingRegressor()
HistGradientBoosting  = HistGradientBoostingRegressor()


# Instantiate the machine learning models with pipelines
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
SupportVectorMachine_pipe = Pipeline([('standardize', StandardScaler()), ('regressor', SupportVectorMachine)])
KernelRidge_pipe          = Pipeline([('standardize', StandardScaler()), ('regressor', KernelRidge)])
MultiLayerPerceptron_pipe = Pipeline([('standardize', StandardScaler()), ('regressor', MultiLayerPerceptron)])
KNeighbors_pipe           = Pipeline([('standardize', StandardScaler()), ('regressor', KNeighbors)])
ExtraTree_pipe            = Pipeline([('standardize', StandardScaler()), ('regressor', ExtraTree)])
DecisionTree_pipe         = Pipeline([('standardize', StandardScaler()), ('regressor', DecisionTree)])
RandomForest_pipe         = Pipeline([('standardize', StandardScaler()), ('regressor', RandomForest)])
GradientBoosting_pipe     = Pipeline([('standardize', StandardScaler()), ('regressor', GradientBoosting)])
HistGradientBoosting_pipe = Pipeline([('standardize', StandardScaler()), ('regressor', HistGradientBoosting)])


print()
print("Default Regressor Hyperparameter values:")
start = time.time()
print(KernelRidge.get_params())
end = time.time()
print("score with default values = ", getDefaultAccuracy(KernelRidge, KernelRidgeParameters, X, y))
print("Time Elapsed = ", end - start)

print()
start = time.time()
#get_GridSearchCV(KernelRidge, KernelRidgeParameters, X, y)
end = time.time()
print("Time Elapsed = ", end - start)

print()
start = time.time()
#get_RandomizedGridSearchCV(KernelRidge, KernelRidgeParameters, X, y)
end = time.time()
print("Time Elapsed = ", end - start)

print()
start = time.time()
get_GeneticGridSearchCV(KernelRidge, KernelRidgeParameters, X, y)
end = time.time()
print("Time Elapsed = ", end - start)


#get_best_model_and_accuracy(SupportVectorMachine, SupportVectorMachineParameters, X, y)
#get_best_model_and_accuracy(KernelRidge,          KernelRidgeParameters,          X, y)
#get_best_model_and_accuracy(MultiLayerPerceptron, MultiLayerPerceptronParameters, X, y)
#get_best_model_and_accuracy(KNeighbors,           KNeighborsParameters,           X, y)
#get_best_model_and_accuracy(ExtraTree,            ExtraTreeParameters,            X, y)
#get_best_model_and_accuracy(DecisionTree,         DecisionTreeParameters,         X, y)
#get_best_model_and_accuracy(RandomForest,         RandomForestParameters,         X, y)
#get_best_model_and_accuracy(GradientBoosting,     GradientBoostingParameters,     X, y)
#get_best_model_and_accuracy(HistGradientBoosting, HistGradientBoostingParameters, X, y)


#get_best_model_and_accuracy(SupportVectorMachine_pipe, SupportVectorMachineParameters_pipe, X, y)
#get_best_model_and_accuracy(KernelRidge_pipe,          KernelRidgeParameters_pipe,          X, y)
#get_best_model_and_accuracy(MultiLayerPerceptron_pipe, MultiLayerPerceptronParameters_pipe, X, y)
#get_best_model_and_accuracy(KNeighbors_pipe,           KNeighborsParameters_pipe,           X, y)
#get_best_model_and_accuracy(ExtraTree_pipe,            ExtraTreeParameters_pipe,            X, y)
#get_best_model_and_accuracy(DecisionTree_pipe,         DecisionTreeParameters_pipe,         X, y)
#get_best_model_and_accuracy(RandomForest_pipe,         RandomForestParameters_pipe,         X, y)
#get_best_model_and_accuracy(GradientBoosting_pipe,     GradientBoostingParameters_pipe,     X, y)
#get_best_model_and_accuracy(HistGradientBoosting_pipe, HistGradientBoostingParameters_pipe, X, y)
