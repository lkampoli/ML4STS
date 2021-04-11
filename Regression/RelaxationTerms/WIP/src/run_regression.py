#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

    parser = argparse.ArgumentParser(description='relaxation terms regression')

#    parser.add_argument('-p', '--process', type=str,
#                        choices=["shear", "bulk", "conductivity", "thermal_diffusion", "mass_diffusion"],
#                        default="shear,bulk,conductivity,thermal_diffusion,mass_diffusion",
#                        help='Comma-separated names of transport properties whose regression is performed')

    parser.add_argument('-a', '--algorithm', type=str,
                        choices=['DT', 'RF', 'ET', 'GP', 'KN', 'SVM', 'KR', 'GB', 'HGB', 'MLP'],
                        default='DT',
                        help='regression algorithm')

    args = parser.parse_args()

#    process   = args.process.split(',')
#    print("Process: ", colored(process[0], 'green'))

    algorithm = args.algorithm.split(',')
    print("Algorithm: ", colored(algorithm[0],'blue'))

    src_dir = "."
    print("SRC: ", colored(src_dir,'yellow'))

    output_dir = src_dir+"/.."
    print("OUTPUT: ", colored(output_dir,'red'))

    n_jobs = 2

    # Import database
    dataset=np.loadtxt("../data/transposed_reshaped_data.txt")
#   with open('../data/TCs_air5.txt') as f:
#       lines = (line for line in f if not line.startswith('#'))
#       dataset = np.loadtxt(lines, skiprows=1)

    print(dataset.shape)

#    if (process[0] == "shear"):
#        x = dataset[:,0:7] # T, P, x_N2, x_O2, x_NO, x_N, x_O
#        y = dataset[:,7:8] # shear viscosity
#    elif (process[0] == "bulk"):
#        x = dataset[:,0:7] # T, P, x_N2, x_O2, x_NO, x_N, x_O
#        y = dataset[:,8:9] # bulk viscosity
#    elif (process[0] == "conductivity"):
#        x = dataset[:,0:7] # T, P, x_N2, x_O2, x_NO, x_N, x_O
#        y = dataset[:,9:10]# thermal conductivity
#    elif (process[0] == "thermal_diffusion"):
#        x = dataset[:,0:7] # T, P, x_N2, x_O2, x_NO, x_N, x_O
#        y = dataset[:,10:] # thermal diffusion, D_Ti
#    elif (process[0] == "mass_diffusion"):
#        x = dataset[:,0:7] # T, P, x_N2, x_O2, x_NO, x_N, x_O
#        y = dataset[:,:]   # mass diffusion TODO

    x = dataset[:,0:50]  # ni_n[47], na_n[1], V, T
    y = dataset[:,50:]   # RD_mol[47], RD_at[1]

    print(x.shape)
    print(y.shape)

    print("### Phase 1: PRE_PROCESSING ###")
    ########################################

    # 1.0) create directory tree
    model, scaler, figure = utils.mk_tree(algorithm[0], output_dir)

    # 1.1) train/test split dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=69)

    # 1.2) scale data and save scalers
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

    dump(sc_x, open(scaler+"/scaler_x.pkl", 'wb'))
    dump(sc_y, open(scaler+"/scaler_y.pkl", 'wb'))

    print("### Phase 2: PROCESSING ###")
    ####################################

    # 2.0) estimator selection
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
    
    # 2.1) search for best hyper-parameters combination
    # Exhaustive search over specified parameter values for the estimator
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    gs = GridSearchCV(est, cv=10, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2',
                      refit=True, pre_dispatch='n_jobs', error_score=np.nan, return_train_score=True)
    
    # Randomized search on hyper parameters
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV
    # class sklearn.model_selection.RandomizedSearchCV(estimator, param_distributions, *, n_iter=10, scoring=None, n_jobs=None, refit=True, 
    #                                                  cv=None, verbose=0, pre_dispatch='2*n_jobs', random_state=None, error_score=nan, 
    #                                                  return_train_score=False)
    #gs = RandomizedSearchCV(est, cv=10, n_iter=10, param_distributions=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2',
    #                        refit=True, pre_dispatch='n_jobs', error_score=np.nan, return_train_score=True)

    # 2.2) training
    utils.fit(x_train, y_train, gs)

    # 2.3) prediction
    y_regr = utils.predict(x_test, gs)

    print("### Phase 3: POST-PROCESSING ###")
    #########################################

    # 3.0) save best hyper-parameters
    results = pd.DataFrame(gs.cv_results_)
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html
    #compression_opts = dict(method='zip', archive_name='GridSearchCV_results.csv')
    #results.to_csv('GridSearchCV_results.zip', index=False, compression=compression_opts)
    results.to_csv(model+"/../"+"GridSearchCV_results.csv", index=False, sep='\t', encoding='utf-8')
    
    # results print screen
    print("Best: %f using %s" % (gs.best_score_, gs.best_params_))
    means  = gs.cv_results_['mean_test_score']
    stds   = gs.cv_results_['std_test_score']
    params = gs.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    # 3.1) compute score metrics
    utils.scores(sc_x, sc_y, x_train, y_train, x_test, y_test, model, gs)

    # 3.2) back to original values (unscaling)
    x_test_dim = sc_x.inverse_transform(x_test)
    y_test_dim = sc_y.inverse_transform(y_test)
    y_regr_dim = sc_y.inverse_transform(y_regr)          

    # 3.3) make plots
    utils.draw_plot(x_test_dim, y_test_dim, y_regr_dim, figure)

    # 3.4) save model to disk
    dump(gs, model+"/model.sav")


if __name__ == "__main__":
    main()