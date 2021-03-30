#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import sys
import argparse
import shutil

import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

from joblib import dump, load

import preprocessing
import DT
import fit
import print_summary
import predict
import mk_plot


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
    #print(args)

    process = args.process.split(',')
    #print(process)

    algorithm = args.algorithm.split(',')
    #print(algorithm)

    if (process[0] == 'DR'):

          print("Process: ", process[0])

          # assign data process directory
          directory = process[0]+'/data/processes'
  
          for filename in os.listdir(directory):
                f = os.path.join(directory, filename)
   
          # checking if it is a file
          if os.path.isfile(f):
                print(f)
      
    # 2 cores
    n_jobs = 2

    parent_dir = "."
    print(parent_dir)

    # 1) Pre-processing phase

    data, dir, proc, model, scaler, figure = preprocessing.mk_tree(f, parent_dir, process[0], algorithm[0])

    print("Loading dataset ...")
    dataset_T = np.loadtxt(parent_dir+"/"+process[0]+"/data/Temperatures.csv")
    dataset_k = np.loadtxt(parent_dir+"/"+process[0]+"/data/"+dir+"/"+proc+"/"+data+".csv")
    print("Loading dataset OK!")
    
    x = dataset_T.reshape(-1,1)
    y = dataset_k[:,:]
   
    print(dataset_T.shape)
    print(dataset_k.shape)

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
     
        est, hyper_params = DT.est_DT()
   
        # Exhaustive search over specified parameter values for the estimator
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
        gs = GridSearchCV(est, cv=10, param_grid=hyper_params, verbose=2, n_jobs=n_jobs, scoring='r2',
                          refit=True, pre_dispatch='n_jobs', error_score=np.nan, return_train_score=True)
    
        fit.fit(x_train,y_train,gs)
        
        y_regr = predict.predict(x_test, gs)
        
        print_summary.scores(sc_x, sc_y, x_train, y_train, x_test, y_test, model, gs)

        x_test_dim = sc_x.inverse_transform(x_test)
        y_test_dim = sc_y.inverse_transform(y_test)
        y_regr_dim = sc_y.inverse_transform(y_regr)          

        mk_plot.draw_plot(x_test_dim, y_test_dim, y_regr_dim, figure, data)

        # save the model to disk
        dump(gs, model+"/model_MO_"+data+'.sav')

if __name__ == "__main__":
    main()
