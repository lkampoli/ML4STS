#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import shutil
from pathlib import Path

from sklearn.metrics import *

import matplotlib.pyplot as plt


def mk_tree(process, algorithm, output_dir):
    
    models  = "models"
    scalers = "scalers"
    figures = "figures"

    model  = os.path.join(output_dir+"/"+process+"/"+algorithm, models)
    scaler = os.path.join(output_dir+"/"+process+"/"+algorithm, scalers)
    figure = os.path.join(output_dir+"/"+process+"/"+algorithm, figures)

    print(model)
    print(scaler)
    print(figure)

    # remove pre-existing directories
    shutil.rmtree(model,  ignore_errors=True)
    shutil.rmtree(scaler, ignore_errors=True)
    shutil.rmtree(figure, ignore_errors=True)

    print("Model: ", model)
    print("Scaler: ", scaler)
    print("Figure: ", figure)

    Path(output_dir+"/"+process+"/"+algorithm).mkdir(parents=True, exist_ok=True)

    os.mkdir(model)
    os.mkdir(scaler)
    os.mkdir(figure)

    print("Directory '%s' created" %models)
    print("Directory '%s' created" %scalers)
    print("Directory '%s' created" %figures)

    return model, scaler, figure


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
   
   with open(data+"/../"+'output.log', 'w') as f:
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
   
    plt.scatter(x_test_dim[:,0], y_test_dim[:], s=5,   c='k', marker='o', label='KAPPA')
    plt.scatter(x_test_dim[:,0], y_regr_dim[:], s=2.5, c='r', marker='o', label='DecisionTree')
    plt.ylabel(r'$\eta$ [Pa·s]')
    plt.xlabel('T [K] ')
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure+"/regression_"+data+'.eps')
    #plt.show()
    plt.close()
