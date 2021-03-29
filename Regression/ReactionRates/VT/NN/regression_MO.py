#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from sklearn import metrics
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard

#from IPython.display import clear_output
#from livelossplot import PlotLossesKeras
#from keras.callbacks import TensorBoard

from keras.utils.vis_utils import plot_model
from keras.models import load_model

#from ann_visualizer.visualize import ann_viz;
from keras.models import model_from_json

from keras_sequential_ascii import keras2ascii
#from livelossplot import PlotLossesKeras

from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras import regularizers

import time
import sys
import os
import shutil

sys.path.insert(0, '../../../../Utilities/')

from plotting import newfig, savefig
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import pandas as pd
import seaborn as sns

import operator
import itertools

from sklearn import metrics
from sklearn.metrics import *

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV

import joblib
from joblib import dump, load
import pickle

from sklearn.inspection import permutation_importance

from sklearn.tree import DecisionTreeRegressor

n_jobs = 2

# Read database filename from command-line input argument
dataset = sys.argv[1]
folder  = dataset[9:14]
process = dataset[18:22]

print(dataset)
print(folder)
print(process)

# Parent Directory path
parent_dir = "./"+dataset

# Directories
models  = "models"
scalers = "scalers"
figures = "figures"

shutil.rmtree(dataset, ignore_errors=True) 
shutil.rmtree(models,  ignore_errors=True) 
shutil.rmtree(scalers, ignore_errors=True) 
shutil.rmtree(figures, ignore_errors=True) 

# Path
model  = os.path.join(parent_dir, models)
scaler = os.path.join(parent_dir, scalers)
figure = os.path.join(parent_dir, figures)

from pathlib import Path
Path("./"+dataset).mkdir(parents=True, exist_ok=True)

os.mkdir(model)
os.mkdir(scaler)
os.mkdir(figure)

print("Directory '%s' created" %models)
print("Directory '%s' created" %scalers)
print("Directory '%s' created" %figures)

# Import database
dataset_T = np.loadtxt("../data/Temperatures.csv")
dataset_k = np.loadtxt("../data/"+folder+"/"+process+"/"+dataset+".csv")

x = dataset_T.reshape(-1,1)
y = dataset_k[:,:]

# summarize the dataset
print("X:", x.shape, "Y:", x.shape)
in_dim = x.shape[1]
out_dim = y.shape[1]

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

dump(sc_x, open(dataset+'/scalers/scaler_x_MO_'+dataset+'.pkl', 'wb'))
dump(sc_y, open(dataset+'/scalers/scaler_y_MO_'+dataset+'.pkl', 'wb'))

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:',   y_train.shape)
print('Testing Features Shape:',  x_test.shape)
print('Testing Labels Shape:',    y_test.shape)


print("[INFO] Model build ...")
model = Sequential()

model.add(Dense(126, input_dim=in_dim, kernel_initializer='normal', activation='relu'))
#model.add(layers.Dropout(0.5))
#model.add(Dense(60, kernel_initializer='normal', activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)))
#model.add(Dense(126, activation='linear'))
# https://www.datatechnotes.com/2019/12/multi-output-regression-example-with.html
model.add(Dense(out_dim, activation='linear'))

#opt = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.01)
opt = keras.optimizers.Adam(learning_rate=0.01)

model.summary()

keras2ascii(model)

# mse:  loss = square(y_true - y_pred)
# mae:  loss = abs(y_true - y_pred)
# mape: loss = 100 * abs(y_true - y_pred) / y_true
# msle: loss = square(log(y_true + 1.) - log(y_pred + 1.))
model.compile(loss='mse', metrics=['mse', 'mae', 'mape', 'msle'], optimizer=opt)

#monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto', restore_best_weights=True)

print("[INFO] training model...")
#history = model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=2, validation_data=(x_test, y_test), callbacks=[PlotLossesKeras()])
history = model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=2, validation_data=(x_test, y_test))

# Plot metrics
print(history.history.keys())

# "Loss"
plt.figure()
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.title('model MSE')
plt.ylabel('mean squared error')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.tight_layout()
plt.savefig("MSE.pdf", dpi=150)
plt.show()
plt.close()

plt.figure()
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('model MAE')
plt.ylabel('mean absolute error')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.tight_layout()
plt.savefig("MAE.pdf", dpi=150)
plt.show()
plt.close()

plt.figure()
plt.plot(history.history['mean_absolute_percentage_error'])
plt.plot(history.history['val_mean_absolute_percentage_error'])
plt.title('model MAPE')
plt.ylabel('mean absolute percentage error')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.tight_layout()
plt.savefig("MAPE.pdf", dpi=150)
plt.show()
plt.close()

plt.figure()
# https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/
plt.style.use("ggplot")
plt.plot(history.history['mean_squared_logarithmic_error'])
plt.plot(history.history['val_mean_squared_logarithmic_error'])
plt.title('model MSLE')
plt.ylabel('mean squared logarithmic error')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.tight_layout()
plt.savefig("MSLE.pdf", dpi=150)
plt.show()
plt.close()


# Predict
print("[INFO] predicting...")
t0 = time.time()
pred = model.predict(x_test)
regr_predict = time.time() - t0
print("Prediction for %d inputs in %.6f s" % (x_test.shape[0], regr_predict))

fig, ax = plt.subplots()
ax.scatter(y_test, pred)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

score = metrics.mean_squared_error(pred, y_test)
print("Final score (MSE): {}".format(score))

score = metrics.mean_absolute_error(pred, y_test)
print("Final score (MAE): {}".format(score))

# Measure RMSE error. RMSE is common for regression.
score = np.sqrt(metrics.mean_squared_error(pred, y_test))
print("Final score (RMSE): {}".format(score))

# Regression chart.
def chart_regression(pred, y, sort=True):
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    if sort:
        t.sort_values(by=['y'], inplace=True)
    plt.plot(t['y'].tolist(), label='expected')
    plt.plot(t['pred'].tolist(), label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.tight_layout()
    plt.savefig("adim_regression.pdf", dpi=150)
    plt.show()
    plt.close()

# Plot the chart
chart_regression(pred.flatten(), y_test)

#dump(model, 'model.sav')
model.save('model.sav')

new_model = tf.keras.models.load_model('model.sav')
new_model.summary()

#t0 = time.time()
#gs.fit(x_train, y_train)
#runtime = time.time() - t0
#print("Training time: %.6f s" % runtime)
#
#train_score_mse  = mean_squared_error(      sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
#train_score_mae  = mean_absolute_error(     sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
#train_score_evs  = explained_variance_score(sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
##train_score_me   = max_error(               sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
##train_score_msle = mean_squared_log_error(  sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
#train_score_r2   = r2_score(                sc_y.inverse_transform(y_train), sc_y.inverse_transform(gs.predict(x_train)))
#
#test_score_mse  = mean_squared_error(      sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
#test_score_mae  = mean_absolute_error(     sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
#test_score_evs  = explained_variance_score(sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
##test_score_me   = max_error(               sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
##test_score_msle = mean_squared_log_error(  sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
#test_score_r2   = r2_score(                sc_y.inverse_transform(y_test), sc_y.inverse_transform(gs.predict(x_test)))
#
#print()
#print("The model performance for training set")
#print("--------------------------------------")
#print('MAE is      {}'.format(train_score_mae ))
#print('MSE is      {}'.format(train_score_mse ))
#print('EVS is      {}'.format(train_score_evs ))
##print('ME is       {}'.format(train_score_me  ))
##print('MSLE is     {}'.format(train_score_msle))
#print('R2 score is {}'.format(train_score_r2  ))
#print()
#print("The model performance for testing set" )
#print("--------------------------------------")
#print('MAE is      {}'.format(test_score_mae ))
#print('MSE is      {}'.format(test_score_mse ))
#print('EVS is      {}'.format(test_score_evs ))
##print('ME is       {}'.format(test_score_me  ))
##print('MSLE is     {}'.format(test_score_msle))
#print('R2 score is {}'.format(test_score_r2  ))
#print()
#print("Best parameters set found on development set:")
#print(gs.best_params_)
#print()
#
## Re-train with best parameters
#regr = DecisionTreeRegressor(**gs.best_params_)
#
#t0 = time.time()
#regr.fit(x_train, y_train)
#regr_fit = time.time() - t0
#print("Re-training time: %.6f s" % regr_fit)
#
#t0 = time.time()
#y_regr = regr.predict(x_test)
#regr_predict = time.time() - t0
#print("Prediction for %d inputs in %.6f s" % (x_test.shape[0], regr_predict))
#
#with open(dataset+'/output.log', 'w') as f:
#    print("Training time: %.6f s"   % regr_fit,     file=f)
#    print("Prediction time: %.6f s" % regr_predict, file=f)
#    print(" ",                                      file=f)
#    print("The model performance for training set", file=f)
#    print("--------------------------------------", file=f)
#    print('MAE is      {}'.format(train_score_mae), file=f)
#    print('MSE is      {}'.format(train_score_mse), file=f)
#    print('EVS is      {}'.format(train_score_evs), file=f)
##    print('ME is       {}'.format(train_score_me),  file=f)
##    print('MSLE is     {}'.format(train_score_msle),file=f)
#    print('R2 score is {}'.format(train_score_r2),  file=f)
#    print(" ",                                      file=f)
#    print("The model performance for testing set",  file=f)
#    print("--------------------------------------", file=f)
#    print('MAE is      {}'.format(test_score_mae),  file=f)
#    print('MSE is      {}'.format(test_score_mse),  file=f)
#    print('EVS is      {}'.format(test_score_evs),  file=f)
##    print('ME is       {}'.format(test_score_me),   file=f)
##    print('MSLE is     {}'.format(test_score_msle), file=f)
#    print('R2 score is {}'.format(test_score_r2),   file=f)
#    print(" ",                                      file=f)
#    print("Best parameters set found on dev set:",  file=f)
#    print(gs.best_params_,                          file=f)
#
#x_test_dim = sc_x.inverse_transform(x_test)
#y_test_dim = sc_y.inverse_transform(y_test)
#y_regr_dim = sc_y.inverse_transform(y_regr)
#
#plt.scatter(x_test_dim, y_test_dim[:,5], s=2, c='k', marker='o', label='Matlab')
#plt.scatter(x_test_dim, y_regr_dim[:,5], s=2, c='purple', marker='+', label='DT, i=5')
#
#plt.scatter(x_test_dim, y_test_dim[:,10], s=2, c='k', marker='o', label='Matlab')
#plt.scatter(x_test_dim, y_regr_dim[:,10], s=2, c='r', marker='+', label='DT, i=10')
#
#plt.scatter(x_test_dim, y_test_dim[:,15], s=2, c='k', marker='o', label='Matlab')
#plt.scatter(x_test_dim, y_regr_dim[:,15], s=2, c='c', marker='+', label='DT, i=15')
#
#plt.scatter(x_test_dim, y_test_dim[:,20], s=2, c='k', marker='o', label='Matlab')
#plt.scatter(x_test_dim, y_regr_dim[:,20], s=2, c='g', marker='+', label='DT, i=20')
#
#plt.scatter(x_test_dim, y_test_dim[:,25], s=2, c='k', marker='o', label='Matlab')
#plt.scatter(x_test_dim, y_regr_dim[:,25], s=2, c='y', marker='+', label='DT, i=25')
#
#plt.scatter(x_test_dim, y_test_dim[:,30], s=2, c='k', marker='o', label='Matlab')
#plt.scatter(x_test_dim, y_regr_dim[:,30], s=2, c='b', marker='+', label='DT, i=30')
#
#plt.scatter(x_test_dim, y_test_dim[:,35], s=2, c='k', marker='o', label='Matlab')
#plt.scatter(x_test_dim, y_regr_dim[:,35], s=2, c='m', marker='+', label='DT, i=35')
#
##plt.ylabel(r'$\eta$ [Pa·s]')
#plt.xlabel('T [K] ')
#plt.legend()
#plt.tight_layout()
#plt.savefig(dataset+'/figures/regression_MO_'+dataset+'.pdf')
##plt.show()
#plt.close()

# save the model to disk
#dump(gs, dataset+'/models/model_MO_'+dataset+'.sav')
