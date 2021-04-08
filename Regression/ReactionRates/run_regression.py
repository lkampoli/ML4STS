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

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

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

    n_jobs = 2

    for f in glob.glob(path):
        #print("{bcolors.OKGREEN}f{bcolors.ENDC}")
        print(colored(f, 'red'))
#FIXME: pd skip the first line 
        dataset_k = pd.read_csv(f, delimiter=",").to_numpy()
        dataset_T = pd.read_csv(parent_dir+"/"+process[0]+"/data/Temperatures.csv").to_numpy()
        
        x = dataset_T.reshape(-1,1)
        y = dataset_k

        print("### Phase 1: PRE_PROCESSING ###")
        ########################################
        data, dir, proc, model, scaler, figure = utils.mk_tree(f, parent_dir, process[0], algorithm[0])

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
    
        # Exhaustive search over specified parameter values for the estimator
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
        gs = GridSearchCV(est, cv=10, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2',
                          refit=True, pre_dispatch='n_jobs', error_score=np.nan, return_train_score=True)
        
        utils.fit(x_train, y_train, gs)

        results = pd.DataFrame(gs.cv_results_)
        # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html
        #compression_opts = dict(method='zip', archive_name='GridSearchCV_results.csv')
        #results.to_csv('GridSearchCV_results.zip', index=False, compression=compression_opts)
        results.to_csv(model+"/../"+"GridSearchCV_results.csv", index=False, sep='\t', encoding='utf-8')
        
        y_regr = utils.predict(x_test, gs)
        
        utils.scores(sc_x, sc_y, x_train, y_train, x_test, y_test, model, gs)

        x_test_dim = sc_x.inverse_transform(x_test)
        y_test_dim = sc_y.inverse_transform(y_test)
        y_regr_dim = sc_y.inverse_transform(y_regr)          

        utils.draw_plot(x_test_dim, y_test_dim, y_regr_dim, figure, data)

        # save the model to disk
        dump(gs, model+"/model_MO_"+data+'.sav')


if __name__ == "__main__":
    main()
