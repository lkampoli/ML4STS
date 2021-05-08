#!/usr/bin/env python
# -*- coding: utf-8 -*-

#from daal4py.sklearn import patch_sklearn
#patch_sklearn()

import os
import time
import sys
import argparse
import shutil
import glob

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, make_scorer
from sklearn.pipeline import Pipeline

from ray.tune.sklearn import TuneGridSearchCV
from ray.tune.schedulers import MedianStoppingRule

from joblib import dump, load

import estimators
import utils

from termcolor import colored
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def main():

    parser = argparse.ArgumentParser(description='reaction rates regression')

    parser.add_argument('-p', '--process', type=str,
                        choices=['DR', 'VT', 'VV', 'VV2', 'ZR'],
                        default='DR,VT,VV,VV2,ZR',
                        help='Comma-separated names of properties whose regression is performed')

    parser.add_argument('-a', '--algorithm', type=str,
                        choices=['DT', 'RF', 'ET', 'GP', 'KN', 'SVM', 'KR', 'GB', 'HGB', 'MLP'],
                        default='DT',
                        help='regression algorithm')

    args = parser.parse_args()

    process   = args.process.split(',')
    directory = process[0]+'/data/processes'
    path      = directory+"/*.csv"
    print("Process: ", colored(process[0], 'green'))

    algorithm = args.algorithm.split(',')
    print("Algorithm: ", colored(algorithm[0],'blue'))

    parent_dir = "."
    print("PWD: ", colored(parent_dir,'yellow'))

    n_jobs = 4

    for f in glob.glob(path):
        #print("{bcolors.OKGREEN}f{bcolors.ENDC}")
        print(colored(f, 'red'))
        dataset_k = pd.read_csv(f, delimiter=",").to_numpy()
        dataset_T = pd.read_csv(parent_dir+"/"+process[0]+"/data/Temperatures.csv").to_numpy()

        x = dataset_T.reshape(-1,1)
        y = dataset_k

        print("### Phase 1: PRE_PROCESSING ###")
        ########################################
        data, dir, proc, model, scaler, figure, outfile = utils.mk_tree(f, parent_dir, process[0], algorithm[0])

        # 3) train/test split dataset
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=69)

        # 4) compute and save scaler
        sc_x = StandardScaler()
        sc_y = StandardScaler()

        sc_x.fit(x_train)
        x_train = sc_x.transform(x_train)
        x_test  = sc_x.transform(x_test)

        sc_y.fit(y_train)
        y_train = sc_y.transform(y_train)
        y_test  = sc_y.transform(y_test)

        print('Training Features Shape:', x_train.shape)
        print('Training Labels Shape:',   y_train.shape)
        print('Testing Features Shape:',  x_test.shape)
        print('Testing Labels Shape:',    y_test.shape)

        dump(sc_x, open(scaler+"/scaler_x_MO_"+data+'.pkl', 'wb'))
        dump(sc_y, open(scaler+"/scaler_y_MO_"+data+'.pkl', 'wb'))

        if (algorithm[0] == 'DT'):
            est, hyper_params = estimators.est_DT()

        elif (algorithm[0] == 'ET'):
            est, hyper_params = estimators.est_ET()

        elif (algorithm[0] == 'SVM'):
            est, hyper_params = estimators.est_SVM()

        elif (algorithm[0] == 'KR'):
            est, hyper_params = estimators.est_KR()

        elif (algorithm[0] == 'KN'):
            est, hyper_params = estimators.est_KN()

        elif (algorithm[0] == 'MLP'):
            est, hyper_params = estimators.est_MLP()

        elif (algorithm[0] == 'GB'):
            est, hyper_params = estimators.est_GB()

        elif (algorithm[0] == 'HGB'):
            est, hyper_params = estimators.est_HGB()

        elif (algorithm[0] == 'RF'):
            est, hyper_params = estimators.est_RF()

        elif (algorithm[0] == 'GB'):
            est, hyper_params = estimators.est_GB()

        else:
            print("Algorithm not implemented ...")

        # https://github.com/ray-project/tune-sklearn
        # https://docs.ray.io/en/latest/tune/api_docs/sklearn.html#tune-sklearn-docs
        # class ray.tune.sklearn.TuneGridSearchCV(estimator, param_grid, early_stopping=None, scoring=None,
        # n_jobs=None, cv=5, refit=True, verbose=0, error_score='raise', return_train_score=False,
        # local_dir='~/ray_results', max_iters=1, use_gpu=False, loggers=None, pipeline_auto_early_stop=True,
        # stopper=None, time_budget_s=None, sk_n_jobs=None)
        #scheduler = MedianStoppingRule(grace_period=10.0)
        #gs = TuneGridSearchCV(est, cv=10, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2',
        #                  refit=True, error_score=np.nan, return_train_score=True)
        #tune_search = TuneSearchCV(clf, parameter_grid, search_optimization="hyperopt", n_trials=3, early_stopping=scheduler, max_iters=10)
        #tune_search.fit(x_train, y_train)


        # Exhaustive search over specified parameter values for the estimator
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
        gs = GridSearchCV(est, cv=5, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2',
                          refit=True, pre_dispatch='n_jobs', error_score=np.nan, return_train_score=True)


        # Randomized search on hyper parameters
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV
        # class sklearn.model_selection.RandomizedSearchCV(estimator, param_distributions, *, n_iter=10, scoring=None, n_jobs=None, refit=True,
        #                                                  cv=None, verbose=0, pre_dispatch='2*n_jobs', random_state=None, error_score=nan,
        #                                                  return_train_score=False)
        #gs = RandomizedSearchCV(est, cv=10, n_iter=10, param_distributions=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2',
        #                        refit=True, pre_dispatch='n_jobs', error_score=np.nan, return_train_score=True)

        # Training
        utils.fit(x_train, y_train, gs, outfile)

        results = pd.DataFrame(gs.cv_results_)
        # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html
        #compression_opts = dict(method='zip', archive_name='GridSearchCV_results.csv')
        #results.to_csv('GridSearchCV_results.zip', index=False, compression=compression_opts)
        results.to_csv(model+"/../"+"GridSearchCV_results.csv", index=False, sep='\t', encoding='utf-8')

        #plt.figure(figsize=(12, 4))
        #for score in ['mean_test_recall', 'mean_test_precision', 'mean_test_min_both']:
        #    plt.plot([_[1] for _ in results['param_class_weight']], results[score], label=score)
        #plt.legend();

        #plt.figure(figsize=(12, 4))
        #for score in ['mean_train_recall', 'mean_train_precision', 'mean_test_min_both']:
        #    plt.scatter(x=[_[1] for _ in results['param_class_weight']], y=results[score.replace('test', 'train')], label=score)
        #plt.legend();

        # summarize results
        print("Best: %f using %s" % (gs.best_score_, gs.best_params_))
        means  = gs.cv_results_['mean_test_score']
        stds   = gs.cv_results_['std_test_score']
        params = gs.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

        # Perform prediction
        y_regr = utils.predict(x_test, gs, outfile)

        # Compute the scores
        utils.scores(sc_x, sc_y, x_train, y_train, x_test, y_test, model, gs, outfile)

        # Transform back
        x_test_dim = sc_x.inverse_transform(x_test)
        y_test_dim = sc_y.inverse_transform(y_test)
        y_regr_dim = sc_y.inverse_transform(y_regr)

        # Make figures
        utils.draw_plot(x_test_dim, y_test_dim, y_regr_dim, figure, data)

        # save the model to disk
        dump(gs, model+"/model_MO_"+data+'.sav')


if __name__ == "__main__":
    main()
