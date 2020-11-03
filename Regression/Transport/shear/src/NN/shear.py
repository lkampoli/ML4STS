#!/usr/bin/env python
# coding: utf-8

# https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
# https://machinelearningmastery.com/autokeras-for-classification-and-regression/
# https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/
# https://machinelearningmastery.com/check-point-deep-learning-models-keras/
# https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/
# https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
# https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/

# https://www.pyimagesearch.com/2019/01/21/regression-with-keras/
# https://www.pluralsight.com/guides/regression-keras
# https://datascienceplus.com/keras-regression-based-neural-networks/
# https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1

# tree --dirsfirst --filelimit 10

import time
import sys
sys.path.insert(0, '../../../../../Utilities/')


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

from pandas import read_csv

# Tensorflow
import tensorflow as tf

import os
os.environ['TF_KERAS'] = '1'

# Keras
#import keras
#from keras import metrics
#from keras.models import Sequential
#from keras.layers import Dense

#from keras.wrappers.scikit_learn import KerasRegressor

from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input

from tensorflow.keras import metrics

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.optimizers import RMSprop

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

n_jobs = -1
trial  = 1

# Import database
# https://pandas.pydata.org/pandas-docs/version/0.25.1/reference/api/pandas.DataFrame.to_numpy.html#pandas.DataFrame.to_numpy
#data = pd.read_fwf("../../../Data/TCs_air5.txt").to_numpy()
#x    = data[:,0:7]
#y    = data[:,7:8]

with open('../../../../Data/TCs_air5.txt') as f:
    lines = (line for line in f if not line.startswith('#'))
    dataset = np.loadtxt(lines, skiprows=1)

#dataset = np.loadtxt("../../../Data/TCs_air5.txt")
x = dataset[:,0:7] # T, P, x_N2, x_O2, x_NO, x_N, x_O
y = dataset[:,7:8] # shear

#dataframe = read_csv("housing.csv", delim_whitespace=True, header=None)
#dataset = dataframe.values

# The data is then split into training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=69)

sc_x = StandardScaler()
sc_y = StandardScaler()

sc_x.fit(x_train)
x_train = sc_x.transform(x_train)
x_test  = sc_x.transform(x_test)

sc_y.fit(y_train)
y_train = sc_y.transform(y_train)
y_test  = sc_y.transform(y_test)

#dump(sc_x, open('../../scaler/scaler_x_shear.pkl', 'wb'))
#dump(sc_y, open('../../scaler/scaler_y_shear.pkl', 'wb'))

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:',   y_train.shape)
print('Testing Features Shape:',  x_test.shape)
print('Testing Labels Shape:',    y_test.shape)

# https://stackoverflow.com/questions/44132652/keras-how-to-perform-a-prediction-using-kerasregressor
# https://www.programcreek.com/python/example/88638/keras.wrappers.scikit_learn.KerasRegressor

opt_SGD = SGD(lr=0.01, momentum=0.9, decay=0.01, nesterov=False)
opt_ADAM = Adam(learning_rate=0.01)
opt_ADAM = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
opt_ADAGRAD = Adagrad()
opt_ADAGRAD = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
opt_ADADELTA = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=100)
opt_RMS = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

#lrate = LearningRateScheduler(step_decay)

#class LossHistory(keras.callbacks.Callback):
#    def on_train_begin(self, logs={}):
#        self.losses = []
#        self.lr = []
#
#    def on_epoch_end(self, batch, logs={}):
#        self.losses.append(logs.get('loss'))
#        self.lr.append(step_decay(len(self.losses)))

#loss_history = LossHistory()
#lrate = LearningRateScheduler(step_decay)
#callbacks_list = [loss_history, lrate]
#history = model.fit(x_train, y_train,
#                    validation_data=(x_test, y_test),
#                    epochs=epochs,
#                    batch_size=batch_size,
#                    callbacks=callbacks_list,
#                    verbose=2)

def exp_decay(epoch):
   initial_lrate = 0.1
   k = 0.1
   lrate = initial_lrate * exp(-k*t)
   return lrate

#lrate = LearningRateScheduler(exp_decay)

## define base model
def baseline_model():

	# create model
	model = Sequential()
	model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))

	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

## define the larger model
def larger_model():

	# create model
	model = Sequential()
	model.add(Dense(7, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(6, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))

        # Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

## define wider model
def wider_model():

	# create model
        #model = models.create_mlp(trainX.shape[1], regress=True)
	model = Sequential()
	model.add(Dense(20, input_dim=7, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))

	# Compile model
        #opt = Adam(lr=1e-3, decay=1e-3 / 200)
	model.compile(loss='mean_squared_error', optimizer='adam') #metrics=['mse','mae']
        # model.compile(loss="mean_absolute_percentage_error", optimizer=opt)
        # model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

        # checkpoint
        #filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
        #filepath="weights.best.hdf5"
        #checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        #callbacks_list = [checkpoint]
	return model

model = Sequential()
model.add(Dense(20, input_dim=7, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))

#model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error','mean_absolute_error'])
#model.compile(loss='mse', optimizer='adam', metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])
#model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape', 'cosine'])
model.compile(loss='mse', optimizer=opt_ADAM, metrics=['mse', 'mae', 'mape'])
#model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 'cosine_proximity'])
#model.compile(loss='mse', optimizer='adam', metrics=[metrics.mean_squared_error, metrics.mean_absolute_error, metrics.mean_absolute_percentage_error, metrics.cosine_proximity])
#model.compile(loss='mse', optimizer='adam', metrics=['msle'])

#t0 = time.time()
# evaluate model with standardized dataset
#estimators = []
#estimators.append(('standardize', StandardScaler()))
#estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=50, batch_size=5, verbose=1)))
#estimators.append(('mlp', KerasRegressor(build_fn=larger_model, epochs=50, batch_size=5, verbose=1)))
#estimators.append(('mlp', KerasRegressor(build_fn=wider_model, epochs=100, batch_size=5, verbose=1)))
# model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, callbacks=callbacks_list, verbose=0)
# model.fit(x=trainX, y=trainY,validation_data=(testX, testY), epochs=200, batch_size=8)
# history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100)
#history = model.fit(x_train, y_train, epochs=150, batch_size=50, verbose=1, validation_split=0.2)
history = model.fit(x_train, y_train, epochs=15, batch_size=50, verbose=1)
#history = model.fit(X, X, epochs=500, batch_size=len(X), verbose=2)
#pipeline = Pipeline(estimators)
#kfold = KFold(n_splits=10)
#results = cross_val_score(pipeline, x, y, cv=kfold)
#print("Wider: %.6f (%.6f) MSE" % (results.mean(), results.std()))
#runtime = time.time() - t0
#print("Complexity and bandwidth selected and model fitted in %.6f s" % runtime)

# plot metrics
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['mean_absolute_percentage_error'])
#plt.plot(history.history['cosine_proximity'])
plt.show()

plt.figure()
plt.plot(history.history['mean_squared_error'])
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.show()

print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['mean_squared_error'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# https://www.tensorflow.org/tutorials/keras/regression?hl=ru

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label = 'Val Error')
  #plt.plot(hist['epoch'], hist['val_mae'], label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error')
  plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'], label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()

#plot_history(history)

# Predition
#y_pr = model.predict(x_test)
#res = metrics.r2_score(y_test, y_pr)
#print(res)

#Xnew = np.array([[40, 0, 26, 9000, 8000]])
#Xnew = np.array([[40, 0, 26, 9000, 8000]])
#Xnew= scaler_x.transform(Xnew)
#ynew= model.predict(Xnew)
#invert normalize
#ynew = scaler_y.inverse_transform(ynew)
#Xnew = scaler_x.inverse_transform(Xnew)
#print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
