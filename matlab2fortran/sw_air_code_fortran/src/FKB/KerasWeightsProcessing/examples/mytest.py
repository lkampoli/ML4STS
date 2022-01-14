import keras
from keras.layers import Dense, Input
from keras.models import Sequential, Model

from tensorflow import set_random_seed

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn import metrics

from keras.utils.vis_utils import plot_model
from keras.models import load_model

from ann_visualizer.visualize import ann_viz;
from keras.models import model_from_json

from keras.optimizers import SGD, Adam, RMSprop, Adagrad
#from keras import regularizers

#import pickle
#from joblib import dump, load

np.random.seed(123); set_random_seed(123)

dataset=np.loadtxt("dataset_STS_kd_kr_N2.txt")
print(dataset.shape)
x = dataset[:,0:1]
y = dataset[:,1:11]

in_dim  = x.shape[1]
out_dim = y.shape[1]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=69)

#sc_x = StandardScaler()
#sc_y = StandardScaler()
#
#sc_x.fit(x_train)
#x_train = sc_x.transform(x_train)
#x_test  = sc_x.transform(x_test)
#
#sc_y.fit(y_train)
#y_train = sc_y.transform(y_train)
#y_test  = sc_y.transform(y_test)
#
#dump(sc_x, open('scaler_x.pkl', 'wb'))
#dump(sc_y, open('scaler_y.pkl', 'wb'))

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:'  , y_train.shape)
print('Testing Features Shape:' , x_test.shape)
print('Testing Labels Shape:'   , y_test.shape)

print(x_train)
print(y_train)

model = Sequential()
model.add(Dense(10, input_dim=1, kernel_initializer='normal', activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='linear'))

opt = keras.optimizers.Adam(learning_rate=0.01, decay=1e-3/100)

model.compile(loss='mse', optimizer='Adam')
model.compile(loss='mse', metrics=['mse', 'mae', 'mape', 'msle'], optimizer=opt)

history = model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=2, validation_data=(x_test, y_test))

from convert_weights import txt_to_h5, h5_to_txt
model_name_h5 = 'simpleNN.h5'
model_name_txt = model_name_h5.replace('h5', 'txt')
model.save(model_name_h5, include_optimizer=False)
h5_to_txt(model_name_h5, model_name_txt)
