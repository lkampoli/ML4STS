#!/usr/bin/env python
# coding: utf-8

from sklearn.metrics import *

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