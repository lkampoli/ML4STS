#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import shutil
from pathlib import Path

from sklearn.metrics import *

import matplotlib.pyplot as plt


def mk_tree(filename, parent_dir, process, algorithm):
    
    if (process == "DR"):    
        data = filename[18:36]
        dir  = data[9:14]  
        proc = data[15:18] # dis, rec
    elif (process == "VT"):    
        data = filename[18:35]
        dir  = data[9:14]  # N2-N2
        proc = data[15:17] # down, up
    elif (process == "VV"):#TODO
        data = filename[18:35]
        dir  = data[9:14]  # N2-N2
        proc = data[15:17] # down, up
    elif (process == "VV2"):#TODO
        data = filename[18:35]
        dir  = data[9:14]  # N2-N2
        proc = data[15:17] # down, up

    print("Dataset: ", data)
    print("Folder: ", dir)
    print("Process: ", proc)
    print("parent_dir: ", parent_dir)

    models  = "models"
    scalers = "scalers"
    figures = "figures"

    model  = os.path.join(parent_dir+"/"+process+"/"+algorithm+"/"+data, models)
    scaler = os.path.join(parent_dir+"/"+process+"/"+algorithm+"/"+data, scalers)
    figure = os.path.join(parent_dir+"/"+process+"/"+algorithm+"/"+data, figures)

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

    Path(parent_dir+"/"+process+"/"+algorithm+"/"+data).mkdir(parents=True, exist_ok=True)

    os.mkdir(model)
    os.mkdir(scaler)
    os.mkdir(figure)

    print("Directory '%s' created" %models)
    print("Directory '%s' created" %scalers)
    print("Directory '%s' created" %figures)

    return data, dir, proc, model, scaler, figure


def fit(x,y,gs):
   t0 = time.time()
   gs.fit(x, y)
   runtime = time.time() - t0
   print("Training time: %.6f s" % runtime)

   return gs


def predict(x_test, gs):

   t0 = time.time()
   y_regr = gs.predict(x_test)
   regr_predict = time.time() - t0
   print("Prediction for %d inputs in %.6f s" % (x_test.shape[0], regr_predict))

   return y_regr


def scores(sc_x, sc_y, x_train, y_train, x_test, y_test, data, gs):

   train_score_mse   = mean_squared_error(      sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
   train_score_mae   = mean_absolute_error(     sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
   train_score_evs   = explained_variance_score(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
   #train_score_me   = max_error(               sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
   #train_score_msle = mean_squared_log_error(  sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
   train_score_r2    = r2_score(                sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
   
   test_score_mse   = mean_squared_error(      sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
   test_score_mae   = mean_absolute_error(     sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
   test_score_evs   = explained_variance_score(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
   #test_score_me   = max_error(               sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
   #test_score_msle = mean_squared_log_error(  sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
   test_score_r2    = r2_score(                sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
   
   
   print()
   print("The model performance for training set")
   print("--------------------------------------")
   print('MAE is      {}'.format(train_score_mae ))
   print('MSE is      {}'.format(train_score_mse ))
   print('EVS is      {}'.format(train_score_evs ))
   #print('ME is      {}'.format(train_score_me  ))
   #print('MSLE is    {}'.format(train_score_msle))
   print('R2 score is {}'.format(train_score_r2  ))
   print()
   print("The model performance for testing set" )
   print("--------------------------------------")
   print('MAE is      {}'.format(test_score_mae ))
   print('MSE is      {}'.format(test_score_mse ))
   print('EVS is      {}'.format(test_score_evs ))
   #print('ME is      {}'.format(test_score_me  ))
   #print('MSLE is    {}'.format(test_score_msle))
   print('R2 score is {}'.format(test_score_r2  ))
   print()
   print("Best parameters found for dev set:")
   print(gs.best_params_)
   print()
   
   with open(data+'/output.log', 'w') as f:
       #print("Training time: %.6f s"   % runtime,      file=f)
       #print("Prediction time: %.6f s" % regr_predict, file=f)
       print(" ",                                      file=f)
       print("The model performance for training set", file=f)
       print("--------------------------------------", file=f)
       print('MAE is      {}'.format(train_score_mae), file=f)
       print('MSE is      {}'.format(train_score_mse), file=f)
       print('EVS is      {}'.format(train_score_evs), file=f)
       #print('ME is      {}'.format(train_score_me),  file=f)
       #print('MSLE is    {}'.format(train_score_msle),file=f)
       print('R2 score is {}'.format(train_score_r2),  file=f)
       print(" ",                                      file=f)
       print("The model performance for testing set",  file=f)
       print("--------------------------------------", file=f)
       print('MAE is      {}'.format(test_score_mae),  file=f)
       print('MSE is      {}'.format(test_score_mse),  file=f)
       print('EVS is      {}'.format(test_score_evs),  file=f)
       #print('ME is      {}'.format(test_score_me),   file=f)
       #print('MSLE is    {}'.format(test_score_msle), file=f)
       print('R2 score is {}'.format(test_score_r2),   file=f)
       print(" ",                                      file=f)
       print("Best parameters found for dev set:",     file=f)
       print(gs.best_params_,                          file=f)


def draw_plot(x_test_dim, y_test_dim, y_regr_dim, figure, data):
   
   plt.scatter(x_test_dim, y_test_dim[:,5], s=2, c='k', marker='o', label='Matlab')
   plt.scatter(x_test_dim, y_regr_dim[:,5], s=2, c='purple', marker='+', label='DT, i=5')

   plt.scatter(x_test_dim, y_test_dim[:,10], s=2, c='k', marker='o', label='Matlab')
   plt.scatter(x_test_dim, y_regr_dim[:,10], s=2, c='r', marker='+', label='DT, i=10')

   plt.scatter(x_test_dim, y_test_dim[:,15], s=2, c='k', marker='o', label='Matlab')
   plt.scatter(x_test_dim, y_regr_dim[:,15], s=2, c='c', marker='+', label='DT, i=15')

   plt.scatter(x_test_dim, y_test_dim[:,20], s=2, c='k', marker='o', label='Matlab')
   plt.scatter(x_test_dim, y_regr_dim[:,20], s=2, c='g', marker='+', label='DT, i=20')

   plt.scatter(x_test_dim, y_test_dim[:,25], s=2, c='k', marker='o', label='Matlab')
   plt.scatter(x_test_dim, y_regr_dim[:,25], s=2, c='y', marker='+', label='DT, i=25')

   plt.scatter(x_test_dim, y_test_dim[:,30], s=2, c='k', marker='o', label='Matlab')
   plt.scatter(x_test_dim, y_regr_dim[:,30], s=2, c='b', marker='+', label='DT, i=30')

   plt.scatter(x_test_dim, y_test_dim[:,35], s=2, c='k', marker='o', label='Matlab')
   plt.scatter(x_test_dim, y_regr_dim[:,35], s=2, c='m', marker='+', label='DT, i=35')

   #plt.ylabel(r'$\eta$ [Pa·s]')
   plt.xlabel('T [K] ')
   plt.legend()
   plt.tight_layout()
   plt.savefig(figure+"/regression_MO_"+data+'.pdf')
   #plt.show()
   plt.close()
