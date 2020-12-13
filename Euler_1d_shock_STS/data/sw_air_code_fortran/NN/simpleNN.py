
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.utils.vis_utils import plot_model
from keras.models import load_model

from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras import regularizers
from keras.layers import Dense
from keras.models import Sequential

import pickle
from joblib import dump, load

import sys
sys.path.insert(0, '../../../utilities/')

from plotting import newfig, savefig

from convert_weights import h5_to_txt
from convert_weights import txt_to_h5

dataset=np.loadtxt("../data/dataset_STS_kd_kr_N2.txt")
x = dataset[:,0:1]
y = dataset[:,1:2]

in_dim  = x.shape[1]
out_dim = y.shape[1]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, test_size=0.25, random_state=69)

sc_x = StandardScaler()
sc_y = StandardScaler()

sc_x.fit(x_train)
x_train = sc_x.transform(x_train)
x_test  = sc_x.transform(x_test)

sc_y.fit(y_train)
y_train = sc_y.transform(y_train)
y_test  = sc_y.transform(y_test)

dump(sc_x, open('scaler_x.pkl', 'wb'))
dump(sc_y, open('scaler_y.pkl', 'wb'))

print('Training Features Shape:', x_train.shape)
print('Training Labels Shape:'  , y_train.shape)
print('Testing Features Shape:' , x_test.shape)
print('Testing Labels Shape:'   , y_test.shape)

model = Sequential()
#model.add(Dense(10, input_dim=in_dim, kernel_initializer='normal', activation='relu'))
model.add(Dense(10, input_dim=1, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, activation='linear'))
#model.add(Dense(out_dim, activation='linear'))

#opt = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.01)
#opt = keras.optimizers.Adam(learning_rate=0.01)
#opt = keras.optimizers.Adam(learning_rate=0.01, decay=1e-3/100)
opt = Adam(learning_rate=0.01, decay=1e-3/100)
#model.summary()

#keras2ascii(model)

model.compile(loss='mse', metrics=['mse', 'mae', 'mape', 'msle'], optimizer=opt)

#history = model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=2, validation_data=(x_test, y_test))

#loss_history = np.array(history)
#np.savetxt("loss_history.txt", loss_history, delimiter=",")

#print(history.history.keys())

## "Loss"
#plt.figure()
#plt.plot(history.history['mean_squared_error'])
#plt.plot(history.history['val_mean_squared_error'])
#plt.title('model MSE')
#plt.ylabel('mean squared error')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper left')
#plt.tight_layout()
#plt.savefig("MSE.pdf", dpi=150)
#plt.show()
#plt.close()
#
#plt.figure()
#plt.plot(history.history['mean_absolute_error'])
#plt.plot(history.history['val_mean_absolute_error'])
#plt.title('model MAE')
#plt.ylabel('mean absolute error')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper left')
#plt.tight_layout()
#plt.savefig("MAE.pdf", dpi=150)
#plt.show()
#plt.close()
#
#plt.figure()
#plt.plot(history.history['mean_absolute_percentage_error'])
#plt.plot(history.history['val_mean_absolute_percentage_error'])
#plt.title('model MAPE')
#plt.ylabel('mean absolute percentage error')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper left')
#plt.tight_layout()
#plt.savefig("MAPE.pdf", dpi=150)
#plt.show()
#plt.close()
#
#plt.figure()
## https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/
#plt.style.use("ggplot")
#plt.plot(history.history['mean_squared_logarithmic_error'])
#plt.plot(history.history['val_mean_squared_logarithmic_error'])
#plt.title('model MSLE')
#plt.ylabel('mean squared logarithmic error')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper left')
#plt.tight_layout()
#plt.savefig("MSLE.pdf", dpi=150)
#plt.show()
#plt.close()
#
## Predict
#print("[INFO] predicting...")
#t0 = time.time()
#pred = model.predict(x_test)
#regr_predict = time.time() - t0
#print("Prediction for %d inputs in %.6f s" % (x_test.shape[0], regr_predict))
#
#fig, ax = plt.subplots()
#ax.scatter(y_test, pred, s=2)
#ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=1)
#ax.set_xlabel('Measured')
#ax.set_ylabel('Predicted')
#plt.show()
#
#score = metrics.mean_squared_error(pred, y_test)
#print("Final score (MSE): {}".format(score))
#
#score = metrics.mean_absolute_error(pred, y_test)
#print("Final score (MAE): {}".format(score))
#
## Measure RMSE error. RMSE is common for regression.
#score = np.sqrt(metrics.mean_squared_error(pred, y_test))
#print("Final score (RMSE): {}".format(score))
#
## Regression chart.
#def chart_regression(pred, y, sort=True):
#    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
#    if sort:
#        t.sort_values(by=['y'], inplace=True)
#    plt.plot(t['y'].tolist(), label='expected')
#    plt.plot(t['pred'].tolist(), label='prediction')
#    plt.ylabel('output')
#    plt.legend()
#    plt.tight_layout()
#    plt.savefig("adim_regression.pdf", dpi=150)
#    plt.show()
#    plt.close()
#
## Plot the chart
#chart_regression(pred.flatten(), y_test)

#model.save('model.sav')
#dump(model, 'model.sav')
#
model_name_h5 = 'NN.h5'
model_name_txt = model_name_h5.replace('h5', 'txt')
model.save(model_name_h5, include_optimizer=False)
h5_to_txt(model_name_h5, model_name_txt)
#plot_model(model, to_file="./model.png", show_shapes=True, show_layer_names=True)
#tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

#new_model = tf.keras.models.load_model('model.sav')
#new_model.summary()
