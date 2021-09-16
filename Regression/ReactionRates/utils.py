#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import shutil
from pathlib import Path

from sklearn.metrics import *

import matplotlib.pyplot as plt


def mk_tree(filename, parent_dir, process, algorithm):

#    if (process == "DR"):
#        data = filename[18:36]
#        dir  = data[9:14]
#        proc = data[15:18] # dis, rec
#    elif (process == "VT"):
#        data = filename[18:35]
#        dir  = data[9:14]  # N2-N2
#        proc = data[15:17] # down, up
#    elif (process == "VV"):#TODO
#        data = filename[18:35]
#        dir  = data[9:14]  # N2-N2
#        proc = data[15:17] # down, up
#    elif (process == "VV2"):#TODO
#        data = filename[18:35]
#        dir  = data[9:14]  # N2-N2
#        proc = data[15:17] # down, up
#    elif (process == "ZR"):#TODO
#        data = filename[18:35]
#        dir  = data[9:14]  # N2-N2
#        proc = data[15:17] # down, up
#    else:
#        print("Process not accounted for ... !")

    proc = "_"
    dir  = "_"

    #import numpy as np
    #dataset_k = np.loadtxt(parent_dir+"/"+process[0]+filename)
    #print(dataset_k)

    # Get the filename only from the initial file path.
    data = os.path.basename(filename)
    data = os.path.splitext(data)[0]
    # Use splitext() to get filename and extension separately.
    #(file, ext) = os.path.splitext(filename)
    #file_name = Path(file_path).stem

    print("Dataset: ",    data      )
    print("Folder: ",     dir       )
    print("Process: ",    proc      )
    print("parent_dir: ", parent_dir)

    models   = "models"
    scalers  = "scalers"
    figures  = "figures"
    outfiles = "outfiles"

    model   = os.path.join(parent_dir+"/"+process+"/"+algorithm+"/"+data, models  )
    scaler  = os.path.join(parent_dir+"/"+process+"/"+algorithm+"/"+data, scalers )
    figure  = os.path.join(parent_dir+"/"+process+"/"+algorithm+"/"+data, figures )
    outfile = os.path.join(parent_dir+"/"+process+"/"+algorithm+"/"+data, outfiles)

    print(model  )
    print(scaler )
    print(figure )
    print(outfile)

    shutil.rmtree(data,    ignore_errors=True)
    shutil.rmtree(model,   ignore_errors=True)
    shutil.rmtree(scaler,  ignore_errors=True)
    shutil.rmtree(figure,  ignore_errors=True)
    shutil.rmtree(outfile, ignore_errors=True)

    print("Model: ",   model  )
    print("Scaler: ",  scaler )
    print("Figure: ",  figure )
    print("Outfile: ", outfile)

    Path(parent_dir+"/"+process+"/"+algorithm+"/"+data).mkdir(parents=True, exist_ok=True)

    os.mkdir(model)
    os.mkdir(scaler)
    os.mkdir(figure)
    os.mkdir(outfile)

    print("Directory '%s' created" %models  )
    print("Directory '%s' created" %scalers )
    print("Directory '%s' created" %figures )
    print("Directory '%s' created" %outfiles)

    return data, dir, proc, model, scaler, figure, outfile


def fit(x, y, gs, outfile):
   t0 = time.time()
   gs.fit(x, y)
   #gs.fit(x, y.ravel())
   runtime = time.time() - t0
   print("Training time: %.6f s" % runtime)

   with open(outfile+'/train_time.log', 'w') as f:
       print("Training time: %.6f s" % runtime, file=f)

   # Write file with training times in append mode
   with open(outfile+"/../../train_times.txt", "a") as train_times_file:
       print(runtime, file=train_times_file)

   return gs


def predict(x_test, gs, outfile):
   t0 = time.time()
   y_regr = gs.predict(x_test)
   regr_predict = time.time() - t0
   print("Prediction for %d inputs in %.6f s" % (x_test.shape[0], regr_predict))

   with open(outfile+'/test_time.log', 'w') as f:
       print("Prediction time: %.6f s" % regr_predict, file=f)

   # Write file with prediction times in append mode
   with open(outfile+"/../../test_times.txt", "a") as test_times_file:
       print(regr_predict, file=test_times_file)

   return y_regr


def scores(input_scaler, output_scaler, x_train, y_train, x_test, y_test, data, gs, outfile):

    if input_scaler is not None and output_scaler is not None:

        train_score_mse   = mean_squared_error(      output_scaler.inverse_transform(y_train), input_scaler.inverse_transform(gs.predict(x_train)))
        train_score_mae   = mean_absolute_error(     output_scaler.inverse_transform(y_train), input_scaler.inverse_transform(gs.predict(x_train)))
        train_score_evs   = explained_variance_score(output_scaler.inverse_transform(y_train), input_scaler.inverse_transform(gs.predict(x_train)))
        #train_score_me   = max_error(               output_scaler.inverse_transform(y_train), input_scaler.inverse_transform(gs.predict(x_train)))
        #train_score_msle = mean_squared_log_error(  output_scaler.inverse_transform(y_train), input_scaler.inverse_transform(gs.predict(x_train)))
        train_score_r2    = r2_score(                output_scaler.inverse_transform(y_train), input_scaler.inverse_transform(gs.predict(x_train)))

        test_score_mse   = mean_squared_error(      output_scaler.inverse_transform(y_test), input_scaler.inverse_transform(gs.predict(x_test)))
        test_score_mae   = mean_absolute_error(     output_scaler.inverse_transform(y_test), input_scaler.inverse_transform(gs.predict(x_test)))
        test_score_evs   = explained_variance_score(output_scaler.inverse_transform(y_test), input_scaler.inverse_transform(gs.predict(x_test)))
        #test_score_me   = max_error(               output_scaler.inverse_transform(y_test), input_scaler.inverse_transform(gs.predict(x_test)))
        #test_score_msle = mean_squared_log_error(  output_scaler.inverse_transform(y_test), input_scaler.inverse_transform(gs.predict(x_test)))
        test_score_r2    = r2_score(                output_scaler.inverse_transform(y_test), input_scaler.inverse_transform(gs.predict(x_test)))

    if input_scaler is None and output_scaler is None:

        train_score_mse   = mean_squared_error(      y_train, gs.predict(x_train))
        train_score_mae   = mean_absolute_error(     y_train, gs.predict(x_train))
        train_score_evs   = explained_variance_score(y_train, gs.predict(x_train))
        #train_score_me   = max_error(               y_train, gs.predict(x_train))
        #train_score_msle = mean_squared_log_error(  y_train, gs.predict(x_train))
        train_score_r2    = r2_score(                y_train, gs.predict(x_train))

        test_score_mse   = mean_squared_error(      y_test, gs.predict(x_test))
        test_score_mae   = mean_absolute_error(     y_test, gs.predict(x_test))
        test_score_evs   = explained_variance_score(y_test, gs.predict(x_test))
        #test_score_me   = max_error(               y_test, gs.predict(x_test))
        #test_score_msle = mean_squared_log_error(  y_test, gs.predict(x_test))
        test_score_r2    = r2_score(                y_test, gs.predict(x_test))

# BUG: the scaler of x_train, x_test should be sc_x (input) and NOT sc_y (output)!
#      That's probably why we get error metrics almost perfect :\
#   train_score_mse   = mean_squared_error(      sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
#   train_score_mae   = mean_absolute_error(     sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
#   train_score_evs   = explained_variance_score(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
#   #train_score_me   = max_error(               sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
#   #train_score_msle = mean_squared_log_error(  sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
#   train_score_r2    = r2_score(                sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
#
#   test_score_mse   = mean_squared_error(      sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
#   test_score_mae   = mean_absolute_error(     sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
#   test_score_evs   = explained_variance_score(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
#   #test_score_me   = max_error(               sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
#   #test_score_msle = mean_squared_log_error(  sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
#   test_score_r2    = r2_score(                sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))

#   train_score_mse   = mean_squared_error(      y_train, sclr.inverse_transform(gs.predict(x_train)))
#   train_score_mae   = mean_absolute_error(     y_train, sclr.inverse_transform(gs.predict(x_train)))
#   train_score_evs   = explained_variance_score(y_train, sclr.inverse_transform(gs.predict(x_train)))
#   #train_score_me   = max_error(               y_train, sclr.inverse_transform(gs.predict(x_train)))
#   #train_score_msle = mean_squared_log_error(  y_train, sclr.inverse_transform(gs.predict(x_train)))
#   train_score_r2    = r2_score(                y_train, sclr.inverse_transform(gs.predict(x_train)))
#
#   test_score_mse   = mean_squared_error(      y_test, sclr.inverse_transform(gs.predict(x_test)))
#   test_score_mae   = mean_absolute_error(     y_test, sclr.inverse_transform(gs.predict(x_test)))
#   test_score_evs   = explained_variance_score(y_test, sclr.inverse_transform(gs.predict(x_test)))
#   #test_score_me   = max_error(               y_test, sclr.inverse_transform(gs.predict(x_test)))
#   #test_score_msle = mean_squared_log_error(  y_test, sclr.inverse_transform(gs.predict(x_test)))
#   test_score_r2    = r2_score(                y_test, sclr.inverse_transform(gs.predict(x_test)))

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

    with open(outfile+'/output.log', 'w') as f:
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

    # Write file with train error metrics in append mode
    with open(outfile+"/../../train_error_metrics.txt", "a") as train_err_file:
        print(train_score_mae, train_score_mse, train_score_evs, train_score_r2, file=train_err_file)

    # Write file with test error metrics in append mode
    with open(outfile+"/../../test_error_metrics.txt", "a") as test_err_file:
        print(test_score_mae, test_score_mse, test_score_evs, test_score_r2, file=test_err_file)

# TODO: make it more general and robust to different max vibr. level
def draw_plot(x_test_dim, y_test_dim, y_regr_dim, figure, data):

   plt.scatter(x_test_dim, y_test_dim[:,5], s=2, c='k',      marker='o', label='Matlab')
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

   #plt.scatter(x_test_dim, y_test_dim[:,35], s=2, c='k', marker='o', label='Matlab')
   #plt.scatter(x_test_dim, y_regr_dim[:,35], s=2, c='m', marker='+', label='DT, i=35')

   #plt.ylabel(r'$\eta$ [Pa·s]')
   plt.xlabel('T [K] ')
   plt.legend()
   plt.tight_layout()
   plt.savefig(figure+"/regression_MO_"+data+'.pdf')
   #plt.show()
   plt.close()


# https://machinelearningmastery.com/prepare-data-machine-learning-python-scikit-learn/
# prepare dataset with input and output scalers, can be none
def scale_dataset(x_train, x_test, y_train, y_test, input_scaler, output_scaler):

    # scale inputs
    if input_scaler is not None:

        # fit scaler
        input_scaler.fit(x_train)

        # transform training dataset
        x_train = input_scaler.transform(x_train)

        # transform test dataset
        x_test = input_scaler.transform(x_test)

    if output_scaler is not None:

        # reshape 1d arrays to 2d arrays
	#y_train = y_train.reshape(len(y_train), 1)
	#y_test  = y_test.reshape(len(y_train), 1)

        # fit scaler on training dataset
        output_scaler.fit(y_train)

        # transform training dataset
        y_train = output_scaler.transform(y_train)

        # transform test dataset
        y_test = output_scaler.transform(y_test)

    return x_train, x_test, y_train, y_test


# inverse transform dataset with input and output scalers, can be none
def scale_back_dataset(x_train, x_test, y_train, y_test, y_regr, input_scaler, output_scaler):

    # scale inputs
    if input_scaler is not None:
        # fit scaler
	# input_scaler.fit(x_train)

	# inverse transform training dataset
	# x_train = input_scaler.transform(x_train)

	# inverse transform test dataset
        x_test = input_scaler.inverse_transform(x_test)

    if output_scaler is not None:

        # reshape 1d arrays to 2d arrays
	# y_train = y_train.reshape(len(y_train), 1)
	# y_test  = y_test.reshape(len(y_train), 1)

	# fit scaler on training dataset
	# output_scaler.fit(y_train)

	# inverse transform training dataset
	# y_train = output_scaler.transform(y_train)

	# inverse transform test dataset
        y_test = output_scaler.inverse_transform(y_test)

        # inverse transform regression values
        y_regr = output_scaler.inverse_transform(y_regr)

    if output_scaler is None:
        y_regr = y_regr

    return x_train, x_test, y_train, y_test, y_regr
