#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import sys
#sys.path.insert(0, './utils/')

import argparse
import shutil

import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

from joblib import dump, load


def mk_tree(filename, parent_dir, process, algorithm):

        data = filename[18:36]
        dir  = data[9:14]
        proc = data[15:18]

        print("Dataset: ", data)
        print("Folder: ", dir)
        print("Process: ", proc)
        print("parent_dir: ", parent_dir)

        models  = "models"
        scalers = "scalers"
        figures = "figures"

        model  = os.path.join(parent_dir+"/"+process+"/"+algorithm, models)
        scaler = os.path.join(parent_dir+"/"+process+"/"+algorithm, scalers)
        figure = os.path.join(parent_dir+"/"+process+"/"+algorithm, figures)

        print(model)
        print(scaler)
        print(figure)

        shutil.rmtree(data,   ignore_errors=True)
        shutil.rmtree(model,  ignore_errors=True)
        shutil.rmtree(scaler, ignore_errors=True)
        shutil.rmtree(figure, ignore_errors=True)

        print("Model: ", model)
        print("Scaler: ", scaler)
        print("Figure: ", figure)

        from pathlib import Path
        Path(parent_dir+"/"+process+"/"+algorithm+"/"+data).mkdir(parents=True, exist_ok=True)

        os.mkdir(model)
        os.mkdir(scaler)
        os.mkdir(figure)

        print("Directory '%s' created" %models)
        print("Directory '%s' created" %scalers)
        print("Directory '%s' created" %figures)

        return data, dir, proc


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

    if (process[0] == 'DR' and algorithm[0] == 'DT'):
          print("Process: ", process[0], "Algorithm: ", algorithm[0])

          # assign directory
          directory = 'DR/data/processes'
  
          for filename in os.listdir(directory):
                f = os.path.join(directory, filename)
   
          # checking if it is a file
          if os.path.isfile(f):
                print(f)
      
          n_jobs = 2

          parent_dir = "."
          print(parent_dir)

          data, dir, proc = mk_tree(f, parent_dir, process[0], algorithm[0])

          print("Loading dataset ...")
          dataset_T = np.loadtxt(parent_dir+"/"+process[0]+"/data/Temperatures.csv")
          dataset_k = np.loadtxt(parent_dir+"/"+process[0]+"/data/"+dir+"/"+proc+"/"+data+".csv")
          print("Loading dataset OK!")
    
          x = dataset_T.reshape(-1,1)
          y = dataset_k[:,:]
   
          print(dataset_T.shape)
          print(dataset_k.shape)

          x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=69)

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

          
if __name__ == "__main__":
    main()
