# https://datascienceplus.com/keras-regression-based-neural-networks/
# https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_04_3_regression.ipynb
# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
# https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
# https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/
# https://stackoverflow.com/questions/49008074/how-to-create-a-neural-network-for-regression
# https://www.pyimagesearch.com/2019/01/21/regression-with-keras/
# https://www.tensorflow.org/tutorials/keras/regression?hl=en
# https://keras.rstudio.com/articles/tutorial_basic_regression.html
# https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network
# https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network
# https://mydeeplearningnb.wordpress.com/2019/02/23/convnet-for-classification-of-cifar-10/

# https://www.omidrouhani.com/research/logisticregression/html/logisticregression.htm
# https://medium.com/@themantalope/glms-cpus-and-gpus-an-introduction-to-machine-learning-through-logistic-regression-python-and-3f226196b1db
# https://developer.nvidia.com/discover/logistic-regression

# https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
# https://towardsdatascience.com/scale-standardize-or-normalize-with-scikit-learn-6ccc7d176a02
# https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/

# https://stackoverflow.com/questions/9622163/save-plot-to-image-file-instead-of-displaying-it-using-matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from sklearn import metrics
from tensorflow import keras
from tensorflow.keras import layers

# it seems that it doesn't help
from tensorflow.keras.callbacks import EarlyStopping

import os;
path="."
os.chdir(path)
os.getcwd()

# Variables
#dataset=np.loadtxt("dataset_MD.csv", delimiter=",")
# Let's consider a smaller dataset, up to 10000 K ...
dataset=np.loadtxt("data/dataset_MD_10000K.csv", delimiter=",")
x=dataset[:,0:4]
y=dataset[:,4] # 0: X, 1: T, 2: I, 3: J, 4: MD (mass diffusion)

# Implementing a NN, it is beneficial to normalize the variables.
# Therefore, variables are transformed using the MaxMinScaler().
print("[INFO] MinMaxScaling data...")
y=np.reshape(y, (-1,1))
scaler_x = MinMaxScaler() #StandardScaler() #RobustScaler() #MaxAbsScaler()
scaler_y = MinMaxScaler()
print(scaler_x.fit(x))
xscale=scaler_x.transform(x)
print(scaler_y.fit(y))
yscale=scaler_y.transform(y)

# The data is then split into training and test data
# Let's use 20% of original data for testing
# The random_state is just for reproducibility (see function doc)
print("[INFO] constructing training/testing split...")
#X_train, X_test, y_train, y_test = train_test_split(xscale, yscale)
X_train, X_test, y_train, y_test = train_test_split(xscale, yscale, test_size=0.2, random_state=42)

model = Sequential()

# Usually it's a good practice to apply following formula in order to find out the total number of hidden layers needed.
#
# Nh = Ns/(α∗ (Ni + No))
#
# where
#
#    Ni = number of input neurons.
#    No = number of output neurons.
#    Ns = number of samples in training data set.
#    α = an arbitrary scaling factor usually 2-10.
#
# ... but I don't care and I take random number of layers and neurons per layer,
# then manually tune the NN.

model.add(Dense(100, input_dim=4, kernel_initializer='normal', activation='relu')) # Hidden 1
#model.add(Dense(100, kernel_initializer='normal', activation='relu'))              # Hidden 2
#model.add(Dense(100, kernel_initializer='normal', activation='relu'))              # Hidden 2
#model.add(Dense(100, kernel_initializer='normal', activation='relu'))              # Hidden 2
#model.add(Dense(100, kernel_initializer='normal', activation='relu'))              # Hidden 2
#model.add(Dense(100, kernel_initializer='normal', activation='relu'))              # Hidden 2
#model.add(Dense(100, kernel_initializer='normal', activation='relu'))              # Hidden 2
#model.add(Dense(100, kernel_initializer='normal', activation='relu'))              # Hidden 2
#model.add(Dense(100, kernel_initializer='normal', activation='relu'))              # Hidden 2
#model.add(Dense(100, kernel_initializer='normal', activation='relu'))              # Hidden 2
model.add(Dense(1, activation='linear'))                                            # Output
model.summary()

# mse:  loss = square(y_true - y_pred)
# mae:  loss = abs(y_true - y_pred)
# mape: loss = 100 * abs(y_true - y_pred) / y_true
# msle: loss = square(log(y_true + 1.) - log(y_pred + 1.))
model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape', 'msle'])

#monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto', restore_best_weights=True)

# Train model
# The validation_split set to 0.2, 80% of the training data is used to test the model, while the remaining 20% is used for testing.
#
# one epoch = one forward pass and one backward pass of all the training examples
#
# batch size = the number of training examples in one forward/backward pass. The higher the batch size, the more memory space.
#
# number of iterations = number of passes, each pass using [batch size] number of examples.
# One pass = one forward pass + one backward pass (we do not count the forward pass and backward pass as two different passes).
#
# The batch size defines the number of samples that will be propagated through the network.
#
# Advantages of using a batch size < number of all samples:
#
# * It requires less memory. Since you train the network using fewer samples, the overall training procedure requires less memory.
#   That's especially important if you are not able to fit the whole dataset in your machine's memory.
#
# * Typically networks train faster with mini-batches. That's because we update the weights after each propagation.
#
# Disadvantages of using a batch size < number of all samples:
#
# * The smaller the batch the less accurate the estimate of the gradient will be.

print("[INFO] training model...")
#history = model.fit(X_train, y_train, epochs=100, batch_size=50, verbose=2, validation_split=0.2)
#history = model.fit(X_train, y_train, epochs=100, batch_size=50, verbose=2, validation_split=0.2, callbacks=[plot_losses])
#history = model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[monitor], verbose=2, batch_size=50, epochs=100)
history = model.fit(X_train, y_train, epochs=10, batch_size=128, verbose=1, validation_data=(X_test, y_test))

#loss_history = np.array(history)
#np.savetxt("loss_history.txt", loss_history, delimiter=",")

# Plot metrics
print(history.history.keys())

# "Loss"
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.savefig('loss.png')

#plt.figure()
#plt.plot(history.history['mean_squared_error'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper left')
#plt.show()
#
#plt.figure()
#plt.plot(history.history['mean_absolute_error'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper left')
#plt.show()
#
#plt.figure()
#plt.plot(history.history['mean_absolute_percentage_error'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'validation'], loc='upper left')
#plt.show()

# Predict
print("[INFO] predicting...")
pred = model.predict(X_test)

# Measure MSE error.
score = metrics.mean_squared_error(pred, y_test)
print("Final score (MSE): {}".format(score))

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
    plt.show()
    plt.savefig('prediction.png')

# Plot the chart
chart_regression(pred.flatten(), y_test)

# Pick up a single value ...
# 0.9, 9000, 35, 15, -0.00401098189571194585
Xnew = np.array([[0.9, 9000, 15, 48]])
Xnew = scaler_x.transform(Xnew)
ynew = model.predict(Xnew)

# Invert normalize
ynew = scaler_y.inverse_transform(ynew)
Xnew = scaler_x.inverse_transform(Xnew)
print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))

plt.scatter(x[2374591:2376991,3], y[2374591:2376991], s=0.5)
plt.plot(x[2376321,1], ynew[0], 'o', color='black')
plt.title(' ')
plt.xlabel(' ')
plt.ylabel(' ')
plt.show()
plt.savefig('single_value.png')
